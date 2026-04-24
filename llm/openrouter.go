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

const openRouterURL = "https://openrouter.ai/api/v1/chat/completions"

// OpenRouterProvider calls the OpenRouter API (OpenAI-compatible).
// Uncensored models like Dolphin are available here with no safety filters.
type OpenRouterProvider struct {
	APIKey       string
	DefaultModel string
	client       *http.Client
}

func (p *OpenRouterProvider) Reset() {} // no per-tick state to clear

func NewOpenRouterProvider(apiKey string) *OpenRouterProvider {
	return &OpenRouterProvider{
		APIKey: apiKey,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *OpenRouterProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (string, error) {
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
		return "", fmt.Errorf("openrouter marshal: %w", err)
	}

	for attempt := range maxRetries {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, openRouterURL, bytes.NewReader(payload))
		if err != nil {
			return "", err
		}
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("HTTP-Referer", "https://lurkr.net")
		req.Header.Set("X-Title", "lurkr arcade")

		resp, err := p.client.Do(req)
		if err != nil {
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("openrouter request error", "attempt", attempt+1, "err", err)
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
				return "", fmt.Errorf("openrouter decode: %w", err)
			}
			if len(result.Choices) == 0 {
				return "", fmt.Errorf("openrouter: no choices in response")
			}
			return result.Choices[0].Message.Content, nil

		case http.StatusUnauthorized, http.StatusForbidden:
			msg := readBody(resp.Body)
			return "", &AuthError{Msg: msg}

		case http.StatusPaymentRequired:
			msg := readBody(resp.Body)
			return "", &BillingError{Msg: "OpenRouter 402: " + truncate(msg, 200)}

		case http.StatusTooManyRequests:
			resp.Body.Close()
			// Rate limit is a quota boundary, not a transient glitch — fail fast
			// so the router falls through to the next provider immediately.
			return "", &RateLimitError{Msg: "openrouter: rate limited — falling through"}

		case 500, 502, 503, 504:
			resp.Body.Close()
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("openrouter server error", "status", resp.StatusCode, "attempt", attempt+1)
				sleep(ctx, backoff)
				continue
			}
			return "", fmt.Errorf("openrouter %d after %d retries", resp.StatusCode, maxRetries)

		default:
			msg := readBody(resp.Body)
			return "", fmt.Errorf("openrouter %d: %s", resp.StatusCode, truncate(msg, 300))
		}
	}
	return "", fmt.Errorf("openrouter: exhausted all retries")
}
