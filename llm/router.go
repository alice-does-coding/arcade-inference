package llm

import (
	"context"
	"log/slog"
)

// Router wraps a primary Provider with an optional fallback.
// On AuthError from the primary, it transparently retries via the fallback
// so the simulation keeps running even when the primary key is invalid.
type Router struct {
	Primary  Provider
	Fallback Provider // nil disables fallback
}

func (r *Router) Chat(ctx context.Context, messages []Message, opts ChatOptions) (string, error) {
	result, err := r.Primary.Chat(ctx, messages, opts)
	if err == nil {
		return result, nil
	}
	if IsAuthError(err) && r.Fallback != nil {
		slog.Warn("primary LLM auth failure — falling back", "err", err)
		return r.Fallback.Chat(ctx, messages, opts)
	}
	return "", err
}

func (r *Router) Reset() {
	r.Primary.Reset()
	if r.Fallback != nil {
		r.Fallback.Reset()
	}
}
