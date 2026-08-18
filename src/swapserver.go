package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
)

// errUnknownModel is returned by an estimator func when the requested
// model isn't in the discovered/configured set.
var errUnknownModel = errors.New("unknown model")

// extractModelID reads the "model" field from an OpenAI/llama.cpp-style
// JSON request body without needing to know the rest of the schema.
func extractModelID(body []byte) (string, error) {
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", fmt.Errorf("invalid JSON body: %w", err)
	}
	if payload.Model == "" {
		return "", fmt.Errorf("missing required \"model\" field")
	}
	return payload.Model, nil
}

// modelEstimator resolves a model ID to its estimated memory footprint at
// the configured context size, or errUnknownModel if it's not registered.
type modelEstimator func(id string) (int64, error)

// newSwapHandler builds the HTTP handler for alpaca swap: every request's
// body is peeked for "model", the supervisor ensures that model is loaded
// (spawning/evicting as needed), and the request is reverse-proxied
// through unchanged — including streaming responses.
func newSwapHandler(sup *supervisor, estimate modelEstimator) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return
		}

		modelID, err := extractModelID(body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		estimatedBytes, err := estimate(modelID)
		if err != nil {
			if errors.Is(err, errUnknownModel) {
				http.Error(w, fmt.Sprintf("unknown model %q", modelID), http.StatusNotFound)
				return
			}
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		target, err := sup.EnsureModel(modelID, estimatedBytes)
		if err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}

		targetURL, err := url.Parse(target)
		if err != nil {
			http.Error(w, "invalid upstream target", http.StatusInternalServerError)
			return
		}

		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))

		proxy := httputil.NewSingleHostReverseProxy(targetURL)
		proxy.FlushInterval = -1 // stream immediately: SSE/chat-completion streaming must not buffer
		proxy.ServeHTTP(w, r)
	})
}
