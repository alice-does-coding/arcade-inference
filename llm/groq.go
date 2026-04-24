package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

const groqURL = "https://api.groq.com/openai/v1/chat/completions"

// GroqProvider calls the Groq API (OpenAI-compatible).
// Groq offers a generous free tier with very fast inference —
// used as a final fallback when all paid providers are dark.
type GroqProvider struct {
	APIKey       string
	DefaultModel string
	client       *http.Client
}

func (p *GroqProvider) Reset() {} // no per-tick state to clear

func NewGroqProvider(apiKey string) *GroqProvider {
	return &GroqProvider{
		APIKey: apiKey,
		client: &http.Client{Timeout: 60 * time.Second},
	}
}

func (p *GroqProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (string, error) {
	model := opts.Model
	if model == "" {
		model = p.DefaultModel
	}

	payload, err := json.Marshal(map[string]any{
		"model":       model,
		"messages":    messages,
		"max_tokens":  opts.MaxTokens,
		"temperature": opts.Temperature,
	})
	if err != nil {
		return "", fmt.Errorf("groq marshal: %w", err)
	}

	for attempt := range maxRetries {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, groqURL, bytes.NewReader(payload))
		if err != nil {
			return "", err
		}
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := p.client.Do(req)
		if err != nil {
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("groq request error", "attempt", attempt+1, "err", err)
				sleep(ctx, backoff)
				continue
			}
			return "", err
		}

		switch resp.StatusCode {
		case http.StatusOK:
			defer resp.Body.Close()
			var result struct {
				Choices []struct {
					Message struct {
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
				return "", fmt.Errorf("groq decode: %w", err)
			}
			if len(result.Choices) == 0 {
				return "", fmt.Errorf("groq: no choices in response")
			}
			return result.Choices[0].Message.Content, nil

		case http.StatusUnauthorized, http.StatusForbidden:
			msg := readBody(resp.Body)
			return "", &AuthError{Msg: msg}

		case http.StatusPaymentRequired:
			msg := readBody(resp.Body)
			return "", &BillingError{Msg: "Groq 402: " + truncate(msg, 200)}

		case http.StatusTooManyRequests:
			resp.Body.Close()
			// Rate limit on a free-tier provider is a per-minute quota, not a
			// transient glitch. Retrying wastes tick time — fail fast so the
			// router can fall through to the next provider immediately.
			return "", &RateLimitError{Msg: "groq: rate limited — falling through"}

		case 500, 502, 503, 504:
			resp.Body.Close()
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("groq server error", "status", resp.StatusCode, "attempt", attempt+1)
				sleep(ctx, backoff)
				continue
			}
			return "", fmt.Errorf("groq %d after %d retries", resp.StatusCode, maxRetries)

		default:
			msg := readBody(resp.Body)
			return "", fmt.Errorf("groq %d: %s", resp.StatusCode, truncate(msg, 300))
		}
	}
	return "", fmt.Errorf("groq: exhausted all retries")
}
