package main

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestExtractModelID_ParsesModelField(t *testing.T) {
	id, err := extractModelID([]byte(`{"model":"muse-glimmer:30b","messages":[]}`))
	if err != nil {
		t.Fatalf("extractModelID: %v", err)
	}
	if id != "muse-glimmer:30b" {
		t.Errorf("id = %q, want %q", id, "muse-glimmer:30b")
	}
}

func TestExtractModelID_ErrorsOnMissingField(t *testing.T) {
	_, err := extractModelID([]byte(`{"messages":[]}`))
	if err == nil {
		t.Fatal("expected error for missing model field")
	}
}

func TestExtractModelID_ErrorsOnInvalidJSON(t *testing.T) {
	_, err := extractModelID([]byte(`not json`))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestSwapHandler_RoutesToResolvedModelAndProxiesResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"echo":"` + string(body) + `"}`))
	}))
	defer upstream.Close()

	sup := newSupervisor(100<<30, 2*time.Second, func(id string, port int) (spawnedChild, error) {
		return &fakeChild{srv: upstream}, nil
	})
	estimator := func(id string) (int64, error) { return 10 << 30, nil }
	handler := newSwapHandler(sup, estimator)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader([]byte(`{"model":"gemma","messages":[]}`)))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if !bytes.Contains(rec.Body.Bytes(), []byte(`"model":"gemma"`)) {
		t.Errorf("expected proxied request body to reach upstream, got %s", rec.Body.String())
	}
}

func TestSwapHandler_RejectsMissingModelField(t *testing.T) {
	sup := newSupervisor(100<<30, 2*time.Second, func(id string, port int) (spawnedChild, error) {
		t.Fatal("should not spawn when model field is missing")
		return nil, nil
	})
	handler := newSwapHandler(sup, func(id string) (int64, error) { return 0, nil })

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestSwapHandler_UnknownModelReturns404(t *testing.T) {
	sup := newSupervisor(100<<30, 2*time.Second, func(id string, port int) (spawnedChild, error) {
		t.Fatal("should not spawn an unknown model")
		return nil, nil
	})
	estimator := func(id string) (int64, error) { return 0, errUnknownModel }
	handler := newSwapHandler(sup, estimator)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"model":"nope"}`)))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}
