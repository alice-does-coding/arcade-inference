package llm

import (
	"context"
	"io"
	"time"
)

// readBody reads and closes the body, returning a string. Failures return "".
func readBody(r io.ReadCloser) string {
	defer r.Close()
	b, _ := io.ReadAll(r)
	return string(b)
}

// truncate returns s[:n] if len(s) > n, otherwise s.
func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// sleep waits for d or until ctx is cancelled.
func sleep(ctx context.Context, d time.Duration) {
	select {
	case <-ctx.Done():
	case <-time.After(d):
	}
}
