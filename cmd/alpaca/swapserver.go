package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
)

// errUnknownModel is returned by an estimator func when the requested
// model isn't in the discovered/configured set.
var errUnknownModel = errors.New("unknown model")

// extractModelID reads the "model" field from a request body: either an
// OpenAI/llama.cpp-style JSON body (chat/completions) or a multipart
// form (audio transcriptions, which must send the file as multipart).
func extractModelID(body []byte, contentType string) (string, error) {
	if mediaType, params, err := mime.ParseMediaType(contentType); err == nil && strings.HasPrefix(mediaType, "multipart/") {
		return extractModelIDFromMultipart(body, params["boundary"])
	}

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

// extractModelIDFromMultipart scans a multipart/form-data body for its
// "model" field without consuming the file part's contents.
func extractModelIDFromMultipart(body []byte, boundary string) (string, error) {
	mr := multipart.NewReader(bytes.NewReader(body), boundary)
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("invalid multipart body: %w", err)
		}
		if part.FormName() == "model" {
			val, err := io.ReadAll(part)
			if err != nil {
				return "", fmt.Errorf("reading model field: %w", err)
			}
			if len(val) == 0 {
				break
			}
			return string(val), nil
		}
	}
	return "", fmt.Errorf("missing required \"model\" field")
}

// modelEstimator resolves a model ID to its estimated memory footprint at
// the configured context size, or errUnknownModel if it's not registered.
type modelEstimator func(id string) (int64, error)

// newUnloadHandler builds the POST /unload endpoint: body {"model": "..."}
// stops that model's running process (if any) so the next request spawns
// a completely fresh one - see supervisor.Evict and docs/known-issues.md
// for why a caller may need this instead of any llama-server cache flag.
func newUnloadHandler(sup *supervisor) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return
		}
		modelID, err := extractModelID(body, r.Header.Get("Content-Type"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		evicted := sup.Evict(modelID)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]bool{"evicted": evicted})
	})
}

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

		modelID, err := extractModelID(body, r.Header.Get("Content-Type"))
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
