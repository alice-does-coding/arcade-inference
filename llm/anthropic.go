package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const anthropicChatURL = "https://api.anthropic.com/v1/messages"
const anthropicVersion = "2023-06-01"

// AnthropicProvider calls the Anthropic Messages API.
type AnthropicProvider struct {
	APIKey       string
	DefaultModel string
	client       *http.Client
}

func NewAnthropicProvider(apiKey string) *AnthropicProvider {
	return &AnthropicProvider{
		APIKey:       apiKey,
		DefaultModel: "claude-sonnet-4-6",
		client:       &http.Client{Timeout: 60 * time.Second},
	}
}

// Reset is a no-op — Anthropic provider has no per-tick state.
func (p *AnthropicProvider) Reset() {}

// Chat sends messages to the Anthropic Messages API and returns the reply.
// System messages are extracted and sent in the top-level system field.
func (p *AnthropicProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (string, error) {
	model := opts.Model
	if model == "" {
		model = p.DefaultModel
	}

	maxTokens := opts.MaxTokens
	if maxTokens == 0 {
		maxTokens = 200
	}

	// Separate system prompt from the conversation.
	var system string
	var msgs []map[string]any
	for _, m := range messages {
		if m.Role == "system" {
			system = m.Content
			continue
		}
		if m.ImageDataURL != "" {
			// Vision message: content is an array of [image block, text block].
			// Parse "data:<mime>;base64,<data>" — fall back to image/jpeg if malformed.
			mediaType := "image/jpeg"
			b64data := m.ImageDataURL
			if comma := strings.Index(m.ImageDataURL, ","); comma >= 0 {
				header := m.ImageDataURL[:comma]
				b64data = m.ImageDataURL[comma+1:]
				if semi := strings.Index(header, ";"); semi >= 0 {
					mediaType = header[5:semi] // strip "data:"
				}
			}
			msgs = append(msgs, map[string]any{
				"role": m.Role,
				"content": []map[string]any{
					{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": mediaType,
							"data":       b64data,
						},
					},
					{"type": "text", "text": m.Content},
				},
			})
		} else {
			msgs = append(msgs, map[string]any{"role": m.Role, "content": m.Content})
		}
	}
	if len(msgs) == 0 {
		return "", fmt.Errorf("Anthropic: no non-system messages")
	}

	// Anthropic temperature is 0–1.
	temp := opts.Temperature
	if temp > 1.0 {
		temp = 1.0
	}

	body := map[string]any{
		"model":       model,
		"max_tokens":  maxTokens,
		"messages":    msgs,
		"temperature": temp,
	}
	if system != "" {
		body["system"] = system
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("Anthropic marshal: %w", err)
	}

	for attempt := range maxRetries {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, anthropicChatURL, bytes.NewReader(payload))
		if err != nil {
			return "", err
		}
		req.Header.Set("x-api-key", p.APIKey)
		req.Header.Set("anthropic-version", anthropicVersion)
		req.Header.Set("content-type", "application/json")

		resp, err := p.client.Do(req)
		if err != nil {
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				sleep(ctx, backoff)
				continue
			}
			return "", fmt.Errorf("Anthropic request: %w", err)
		}

		switch resp.StatusCode {
		case http.StatusOK:
			return p.decodeChat(resp.Body)

		case http.StatusUnauthorized:
			msg := readBody(resp.Body)
			return "", &AuthError{Msg: msg}

		case http.StatusTooManyRequests:
			resp.Body.Close()
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				sleep(ctx, backoff)
				continue
			}
			return "", &RateLimitError{Msg: fmt.Sprintf("Anthropic rate limit exhausted after %d attempts", maxRetries)}

		case 500, 502, 503, 504:
			resp.Body.Close()
			if attempt < maxRetries-1 {
				backoff := time.Duration(1<<attempt) * time.Second
				sleep(ctx, backoff)
				continue
			}
			return "", fmt.Errorf("Anthropic %d server error after %d retries", resp.StatusCode, maxRetries)

		default:
			msg := readBody(resp.Body)
			return "", fmt.Errorf("Anthropic %d: %s", resp.StatusCode, truncate(msg, 300))
		}
	}
	return "", fmt.Errorf("Anthropic: exhausted all retries")
}

func (p *AnthropicProvider) decodeChat(body io.ReadCloser) (string, error) {
	defer body.Close()
	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	b, _ := io.ReadAll(body)
	if err := json.Unmarshal(b, &result); err != nil {
		return "", fmt.Errorf("Anthropic decode: %w", err)
	}
	for _, c := range result.Content {
		if c.Type == "text" {
			return c.Text, nil
		}
	}
	return "", fmt.Errorf("Anthropic: no text content in response")
}
