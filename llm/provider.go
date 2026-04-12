// Package llm provides a provider-agnostic interface for LLM chat calls.
// All simulation code should call these types instead of touching provider
// SDKs or HTTP clients directly.
package llm

import (
	"context"
	"errors"
	"fmt"
)

// Message is a single turn in a chat conversation.
// ImageDataURL is optional — a base64 data URI (e.g. "data:image/jpeg;base64,...")
// that the provider attaches as an image block before the text Content.
// Only providers that support vision will act on it; others ignore it.
type Message struct {
	Role         string `json:"role"`
	Content      string `json:"content"`
	ImageDataURL string `json:"image_data_url,omitempty"`
}

// ChatOptions controls generation parameters for a single call.
type ChatOptions struct {
	Model       string
	MaxTokens   int
	Temperature float64
}

// Provider is the interface all LLM backends must implement.
type Provider interface {
	// Chat sends messages and returns the assistant reply as a plain string.
	Chat(ctx context.Context, messages []Message, opts ChatOptions) (string, error)
	// Reset clears per-tick state (auth latches, call stats). Call at tick start.
	Reset()
}

// AuthError is returned on 401/403. The caller should stop the run.
type AuthError struct {
	Msg string
}

func (e *AuthError) Error() string { return fmt.Sprintf("LLM auth error: %s", e.Msg) }

// RateLimitError is returned when all retries on 429 are exhausted.
type RateLimitError struct {
	Msg string
}

func (e *RateLimitError) Error() string { return fmt.Sprintf("LLM rate limit: %s", e.Msg) }

// IsAuthError reports whether err is an AuthError.
func IsAuthError(err error) bool {
	var e *AuthError
	return errors.As(err, &e)
}

// IsRateLimitError reports whether err is a RateLimitError.
func IsRateLimitError(err error) bool {
	var e *RateLimitError
	return errors.As(err, &e)
}
