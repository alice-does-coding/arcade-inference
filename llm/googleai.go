package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

const googleAIBaseURL = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"

// GoogleAIProvider calls the Google AI Studio native API.
// Uses x-goog-api-key authentication — works with all AI Studio key formats (AIza… and AQ.…).
// Free tier: 15 RPM, 1M tokens/day for Gemma models. No billing account required.
type GoogleAIProvider struct {
	APIKey       string
	DefaultModel string
	client       *http.Client
}

func (p *GoogleAIProvider) Reset() {}

func NewGoogleAIProvider(apiKey string) *GoogleAIProvider {
	return &GoogleAIProvider{
		APIKey: apiKey,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

// geminiContent is a single turn in Gemini's native format.
type geminiContent struct {
	Role  string         `json:"role"`
	Parts []geminiPart   `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

func (p *GoogleAIProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (string, error) {
	model := opts.Model
	if model == "" {
		model = p.DefaultModel
	}

	// Gemma models on the native API don't support system_instruction.
	// Collect system messages and prepend them to the first user turn.
	var systemParts []string
	var contents []geminiContent
	for _, m := range messages {
		switch m.Role {
		case "system":
			systemParts = append(systemParts, m.Content)
		case "assistant":
			contents = append(contents, geminiContent{
				Role:  "model",
				Parts: []geminiPart{{Text: m.Content}},
			})
		default: // "user"
			contents = append(contents, geminiContent{
				Role:  "user",
				Parts: []geminiPart{{Text: m.Content}},
			})
		}
	}

	// Prepend collected system text to the first user turn.
	if len(systemParts) > 0 && len(contents) > 0 {
		systemText := strings.Join(systemParts, "\n\n")
		for i, c := range contents {
			if c.Role == "user" {
				contents[i].Parts[0].Text = systemText + "\n\n" + c.Parts[0].Text
				break
			}
		}
	}

	body := map[string]any{
		"contents": contents,
		"generationConfig": map[string]any{
			"maxOutputTokens": opts.MaxTokens,
			"temperature":     opts.Temperature,
		},
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("googleai marshal: %w", err)
	}

	url := fmt.Sprintf(googleAIBaseURL, model)

	for attempt := range maxRetries {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return "", err
		}
		req.Header.Set("x-goog-api-key", p.APIKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := p.client.Do(req)
		if err != nil {
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("googleai request error", "attempt", attempt+1, "err", err)
				sleep(ctx, backoff)
				continue
			}
			return "", err
		}

		switch resp.StatusCode {
		case http.StatusOK:
			defer resp.Body.Close()
			var result struct {
				Candidates []struct {
					Content struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					} `json:"content"`
				} `json:"candidates"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
				return "", fmt.Errorf("googleai decode: %w", err)
			}
			if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
				return "", fmt.Errorf("googleai: no content in response")
			}
			// Join parts (usually just one) into a single string.
			var sb strings.Builder
			for _, part := range result.Candidates[0].Content.Parts {
				sb.WriteString(part.Text)
			}
			return sb.String(), nil

		case http.StatusUnauthorized, http.StatusForbidden:
			msg := readBody(resp.Body)
			return "", &AuthError{Msg: msg}

		case http.StatusPaymentRequired:
			// Daily quota exhausted — short cooldown, resets at midnight PT.
			msg := readBody(resp.Body)
			return "", &RateLimitError{Msg: "googleai 402: " + truncate(msg, 200)}

		case http.StatusTooManyRequests:
			resp.Body.Close()
			return "", &RateLimitError{Msg: "googleai: rate limited — falling through"}

		case 500, 502, 503, 504:
			resp.Body.Close()
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("googleai server error", "status", resp.StatusCode, "attempt", attempt+1)
				sleep(ctx, backoff)
				continue
			}
			return "", fmt.Errorf("googleai %d after %d retries", resp.StatusCode, maxRetries)

		default:
			msg := readBody(resp.Body)
			return "", fmt.Errorf("googleai %d: %s", resp.StatusCode, truncate(msg, 300))
		}
	}
	return "", fmt.Errorf("googleai: exhausted all retries")
}
