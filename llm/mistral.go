package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

const mistralChatURL = "https://api.mistral.ai/v1/chat/completions"

// MistralStats holds per-tick call metrics. Read via Stats().
type MistralStats struct {
	Calls        int
	ThrottleTime time.Duration
	APITime      time.Duration
}

// MistralProvider calls the Mistral AI API (OpenAI-compatible endpoint).
//
// Auth latch: on first 401 this tick, subsequent callers bail without making
// HTTP calls. The latch is only cleared every RetryInterval Reset() calls so
// a bad key doesn't waste a call every tick.
type MistralProvider struct {
	APIKey        string
	DefaultModel  string  // used when ChatOptions.Model is empty
	RateLimit     float64 // requests per second; default 2
	RetryInterval int     // clear auth latch every N resets; default 12

	mu            sync.Mutex
	rlNext        time.Time
	authFailed    bool
	resetCount    int
	stats         MistralStats
	client        *http.Client
}

func NewMistralProvider(apiKey string, rateLimit float64) *MistralProvider {
	if rateLimit <= 0 {
		rateLimit = 2
	}
	return &MistralProvider{
		APIKey:        apiKey,
		RateLimit:     rateLimit,
		RetryInterval: 12,
		client:        &http.Client{Timeout: 60 * time.Second},
	}
}

// Reset clears per-tick call stats. The auth latch is only cleared every
// RetryInterval calls so a bad Mistral key doesn't waste a call every tick.
func (p *MistralProvider) Reset() {
	p.mu.Lock()
	p.stats = MistralStats{}
	p.resetCount++
	if p.resetCount%p.RetryInterval == 0 {
		p.authFailed = false
	}
	p.mu.Unlock()
}

// Stats returns a snapshot of per-tick call metrics.
func (p *MistralProvider) Stats() MistralStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stats
}

// Chat calls the Mistral API. Returns the assistant reply as a plain string.
// Retries on 5xx/429/timeout with exponential backoff (up to 6 attempts).
// Raises AuthError on 401. Raises RateLimitError if all 429 retries exhausted.
func (p *MistralProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (string, error) {
	p.mu.Lock()
	if p.authFailed {
		p.mu.Unlock()
		return "", &AuthError{Msg: "Mistral API key invalid (auth failure already seen this tick)"}
	}
	p.mu.Unlock()

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
		return "", fmt.Errorf("Mistral marshal: %w", err)
	}

	for attempt := range maxRetries {
		throttleWait := p.throttle()

		apiStart := time.Now()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, mistralChatURL, bytes.NewReader(payload))
		if err != nil {
			return "", err
		}
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := p.client.Do(req)
		apiElapsed := time.Since(apiStart)

		if err != nil {
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("Mistral request error", "attempt", attempt+1, "backoff", backoff, "err", err)
				sleep(ctx, backoff)
				continue
			}
			return "", err
		}

		switch resp.StatusCode {
		case http.StatusOK:
			content, err := p.decodeChat(resp.Body, opts.Model)
			if err != nil {
				return "", err
			}
			p.mu.Lock()
			p.stats.Calls++
			p.stats.ThrottleTime += throttleWait
			p.stats.APITime += apiElapsed
			p.mu.Unlock()
			return content, nil

		case http.StatusUnauthorized:
			msg := readBody(resp.Body)
			p.mu.Lock()
			if !p.authFailed {
				slog.Error("Mistral 401 — invalid or exhausted API key")
				p.authFailed = true
			}
			p.mu.Unlock()
			return "", &AuthError{Msg: msg}

		case http.StatusPaymentRequired:
			msg := readBody(resp.Body)
			return "", &BillingError{Msg: "Mistral 402: " + truncate(msg, 200)}

		case http.StatusTooManyRequests:
			resp.Body.Close()
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("Mistral 429 rate limit", "attempt", attempt+1, "backoff", backoff)
				sleep(ctx, backoff)
				continue
			}
			return "", &RateLimitError{Msg: fmt.Sprintf("Mistral rate limit exhausted after %d attempts", maxRetries)}

		case 500, 502, 503, 504:
			resp.Body.Close()
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("Mistral server error", "status", resp.StatusCode, "attempt", attempt+1, "backoff", backoff)
				sleep(ctx, backoff)
				continue
			}
			return "", fmt.Errorf("Mistral %d server error after %d retries", resp.StatusCode, maxRetries)

		default:
			msg := readBody(resp.Body)
			return "", fmt.Errorf("Mistral %d for %s: %s", resp.StatusCode, opts.Model, truncate(msg, 300))
		}
	}
	return "", fmt.Errorf("Mistral: exhausted all retries")
}

// throttle blocks until the rate-limit gate allows the next call.
// Returns the actual time spent waiting.
func (p *MistralProvider) throttle() time.Duration {
	p.mu.Lock()
	now := time.Now()
	var waited time.Duration
	if p.rlNext.After(now) {
		waited = p.rlNext.Sub(now)
		time.Sleep(waited)
	}
	p.rlNext = time.Now().Add(time.Duration(float64(time.Second) / p.RateLimit))
	p.mu.Unlock()
	return waited
}

func (p *MistralProvider) decodeChat(body io.ReadCloser, model string) (string, error) {
	defer body.Close()
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(body).Decode(&result); err != nil {
		return "", fmt.Errorf("Mistral decode error: %w", err)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("Mistral: no choices in response for %s", model)
	}
	return result.Choices[0].Message.Content, nil
}
