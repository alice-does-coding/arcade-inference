package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	hfChatURL   = "https://router.huggingface.co/v1/chat/completions"
	hfAvatarURL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
	maxRetries  = 6
)

// HFProvider calls an OpenAI-compatible chat-completions endpoint.
//
// Despite the historic name, this client is provider-agnostic: BaseChatURL
// can point at HuggingFace's router OR at a self-hosted llama.cpp / vLLM
// server (e.g. http://garden-arcade-text.flycast:8080). When BaseChatURL
// is empty, defaults to HF for backward compatibility.
//
// Rate limiting: a proactive token-bucket gate serialises concurrent callers.
// Default 8 req/s; override per-host as appropriate.
//
// Auth latch: on first 401 this tick, all subsequent callers bail immediately
// without making HTTP calls. Cleared by Reset().
type HFProvider struct {
	APIKey       string
	BaseChatURL  string  // OAI-compatible /v1/chat/completions endpoint; empty = HF router
	DefaultModel string  // used when ChatOptions.Model is empty
	RateLimit    float64 // requests per second; default 8

	mu         sync.Mutex
	rlNext     time.Time
	authFailed bool
	client     *http.Client
}

func NewHFProvider(apiKey string, rateLimit float64) *HFProvider {
	if rateLimit <= 0 {
		rateLimit = 8
	}
	return &HFProvider{
		APIKey:    apiKey,
		RateLimit: rateLimit,
		client:    &http.Client{Timeout: 120 * time.Second},
	}
}

// NewSelfHostedProvider creates a provider for a self-hosted OpenAI-compatible
// inference server (e.g. llama.cpp's /v1/chat/completions). No API key required;
// auth is the network boundary (flycast / VPC / Tailscale).
//
// baseURL should point at the server root (without /v1/...) — e.g.
// "http://garden-arcade-text.flycast:8080".
//
// Timeout is generous (5 min) because self-hosted CPU inference can queue
// behind concurrent slots — the call is "slow but eventually succeeds." A
// short timeout here means the response arrives after we've disconnected
// and gets thrown away.
func NewSelfHostedProvider(baseURL string) *HFProvider {
	return &HFProvider{
		BaseChatURL: baseURL + "/v1/chat/completions",
		RateLimit:   16, // self-hosted has no upstream rate limit; just throttle a bit
		client:      &http.Client{Timeout: 300 * time.Second},
	}
}

// chatURL returns the configured chat endpoint, defaulting to HF's router
// when BaseChatURL is empty.
func (p *HFProvider) chatURL() string {
	if p.BaseChatURL != "" {
		return p.BaseChatURL
	}
	return hfChatURL
}

// injectNoThink suppresses Qwen3's default thinking mode by ensuring the
// /no_think directive appears in a system message. Qwen3-family models
// recognise this as a soft switch and skip the <think>...</think> block,
// which is critical for chat latency and often pure noise in tiny models.
//
// Only applied when calling a self-hosted endpoint (BaseChatURL set) — HF
// router calls go to whichever model is selected and may not be Qwen.
func injectNoThink(messages []Message) []Message {
	out := make([]Message, len(messages))
	copy(out, messages)
	for i, m := range out {
		if m.Role == "system" {
			if !strings.Contains(m.Content, "/no_think") {
				out[i].Content = "/no_think " + m.Content
			}
			return out
		}
	}
	// No system message — add one.
	return append([]Message{{Role: "system", Content: "/no_think"}}, out...)
}

// Reset clears the auth latch. Call at tick start.
func (p *HFProvider) Reset() {
	p.mu.Lock()
	p.authFailed = false
	p.mu.Unlock()
}

// Chat calls the HF Inference API. Returns the assistant reply as a plain string.
// Retries on 5xx/429 with exponential backoff (up to 6 attempts).
// Raises AuthError on 401/403. Raises RateLimitError if all 429 retries exhausted.
func (p *HFProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (string, error) {
	p.mu.Lock()
	if p.authFailed {
		p.mu.Unlock()
		return "", &AuthError{Msg: "HF API key invalid (auth failure already seen this tick)"}
	}
	p.mu.Unlock()

	model := opts.Model
	if model == "" {
		model = p.DefaultModel
	}
	// Self-hosted callers get /no_think injected by default — see injectNoThink.
	if p.BaseChatURL != "" {
		messages = injectNoThink(messages)
	}
	payload, err := json.Marshal(map[string]any{
		"model":       model,
		"messages":    messages,
		"max_tokens":  opts.MaxTokens,
		"temperature": opts.Temperature,
	})
	if err != nil {
		return "", fmt.Errorf("HF marshal: %w", err)
	}

	for attempt := range maxRetries {
		p.throttle()

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.chatURL(), bytes.NewReader(payload))
		if err != nil {
			return "", err
		}
		// Self-hosted endpoints don't need auth (network boundary IS the auth).
		// Only set Authorization when we have a key — avoids sending "Bearer " to
		// llama.cpp which logs noise for empty-token requests.
		if p.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+p.APIKey)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := p.client.Do(req)
		if err != nil {
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("HF request error", "attempt", attempt+1, "backoff", backoff, "err", err)
				sleep(ctx, backoff)
				continue
			}
			return "", err
		}

		switch resp.StatusCode {
		case http.StatusOK:
			return p.decodeChat(resp.Body, opts.Model)

		case http.StatusUnauthorized:
			msg := readBody(resp.Body)
			p.mu.Lock()
			if !p.authFailed {
				slog.Error("HF 401 — invalid or exhausted API key")
				p.authFailed = true
			}
			p.mu.Unlock()
			return "", &AuthError{Msg: msg}

		case http.StatusForbidden:
			msg := fmt.Sprintf("model access denied for %s — accept terms on huggingface.co: %s", opts.Model, readBody(resp.Body))
			slog.Error("HF 403", "msg", msg)
			return "", &AuthError{Msg: msg}

		case http.StatusPaymentRequired:
			msg := readBody(resp.Body)
			return "", &BillingError{Msg: "HuggingFace 402: " + truncate(msg, 200)}

		case http.StatusBadRequest, 422:
			msg := readBody(resp.Body)
			return "", fmt.Errorf("HF %d bad request for %s: %s", resp.StatusCode, opts.Model, truncate(msg, 300))

		case http.StatusTooManyRequests:
			resp.Body.Close()
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("HF 429 rate limit", "attempt", attempt+1, "backoff", backoff)
				sleep(ctx, backoff)
				continue
			}
			return "", &RateLimitError{Msg: fmt.Sprintf("exhausted after %d attempts", maxRetries)}

		case 500, 502, 503, 504:
			resp.Body.Close()
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				slog.Warn("HF server error", "status", resp.StatusCode, "attempt", attempt+1, "backoff", backoff)
				sleep(ctx, backoff)
				continue
			}
			return "", fmt.Errorf("HF %d server error after %d retries", resp.StatusCode, maxRetries)

		default:
			msg := readBody(resp.Body)
			return "", fmt.Errorf("HF %d for %s: %s", resp.StatusCode, opts.Model, truncate(msg, 300))
		}
	}
	return "", fmt.Errorf("HF: exhausted all retries")
}

// GenerateAvatar generates a 256×256 pixel-art portrait from a bio via FLUX.1-schnell.
// Returns a base64 data URL (data:image/...;base64,...) or empty string on failure.
// Failures are logged but never returned as errors — avatar generation is best-effort.
func (p *HFProvider) GenerateAvatar(ctx context.Context, bio, name string) string {
	subject := bio
	if name != "" {
		subject = name + " — " + truncate(bio, 120)
	} else {
		subject = truncate(bio, 180)
	}
	prompt := "Generate an 8-bit pixel art portrait profile picture based on this description: " + subject +
		". The style should be highly pixelated, with visible grid lines, limited color palette, " +
		"and sharp, blocky edges. Emphasize the retro video game aesthetic, using large, distinct pixels. " +
		"The image should look like it belongs in an early 8-bit or 16-bit era game, " +
		"with minimal detail and exaggerated features."

	payload, _ := json.Marshal(map[string]any{
		"inputs": prompt,
		"parameters": map[string]any{
			"width": 256, "height": 256, "num_inference_steps": 4,
		},
	})

	for attempt := range 4 {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, hfAvatarURL, bytes.NewReader(payload))
		if err != nil {
			return ""
		}
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := p.client.Do(req)
		if err != nil {
			slog.Warn("FLUX avatar timeout", "attempt", attempt+1)
			continue
		}

		switch resp.StatusCode {
		case http.StatusOK:
			data, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				return ""
			}
			mime := resp.Header.Get("Content-Type")
			if mime == "" {
				mime = "image/png"
			}
			return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)

		case http.StatusServiceUnavailable:
			// Model loading — HF returns estimated_time in body
			var body struct {
				EstimatedTime float64 `json:"estimated_time"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&body); err != nil || body.EstimatedTime == 0 {
				body.EstimatedTime = 20
			}
			resp.Body.Close()
			wait := time.Duration(min(body.EstimatedTime, 30)) * time.Second
			slog.Info("FLUX model loading", "wait", wait)
			sleep(ctx, wait)
			continue

		default:
			msg := readBody(resp.Body)
			slog.Warn("FLUX avatar failed", "status", resp.StatusCode, "body", truncate(msg, 200))
			return ""
		}
	}
	slog.Warn("FLUX avatar: all retries exhausted")
	return ""
}

// throttle blocks until the rate-limit gate allows the next call.
// All concurrent goroutines are serialised through this lock — matching Python behaviour.
func (p *HFProvider) throttle() {
	p.mu.Lock()
	now := time.Now()
	if p.rlNext.After(now) {
		time.Sleep(p.rlNext.Sub(now))
	}
	p.rlNext = time.Now().Add(time.Duration(float64(time.Second) / p.RateLimit))
	p.mu.Unlock()
}

func (p *HFProvider) decodeChat(body io.ReadCloser, model string) (string, error) {
	defer body.Close()
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(body).Decode(&result); err != nil {
		return "", fmt.Errorf("HF decode error: %w", err)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("HF: no choices in response for %s", model)
	}
	return result.Choices[0].Message.Content, nil
}
