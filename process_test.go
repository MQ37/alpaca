package main

import (
	"net/http"
	"net/http/httptest"
	"os/exec"
	"testing"
	"time"
)

func TestChildProcess_StartAndStop(t *testing.T) {
	cp := &childProcess{}
	if err := cp.Start(exec.Command("sleep", "30")); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if cp.cmd.Process == nil {
		t.Fatal("expected a running process")
	}

	if err := cp.Stop(2 * time.Second); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if cp.cmd.ProcessState == nil {
		t.Fatal("expected process to have exited after Stop")
	}
}

func TestChildProcess_StopForceKillsIfUnresponsive(t *testing.T) {
	// "sleep" ignores nothing special, but SIGTERM does terminate it normally.
	// Use a very short grace period to exercise the force-kill path without
	// actually needing an unkillable process.
	cp := &childProcess{}
	if err := cp.Start(exec.Command("sleep", "30")); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := cp.Stop(1 * time.Millisecond); err != nil {
		t.Fatalf("Stop: %v", err)
	}
}

func TestWaitHealthy_SucceedsOnce200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := waitHealthy(srv.URL+"/health", 2*time.Second); err != nil {
		t.Fatalf("waitHealthy: %v", err)
	}
}

func TestWaitHealthy_EventuallySucceedsAfterInitialFailures(t *testing.T) {
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := waitHealthy(srv.URL+"/health", 2*time.Second); err != nil {
		t.Fatalf("waitHealthy: %v", err)
	}
	if attempts < 3 {
		t.Errorf("attempts = %d, want >= 3", attempts)
	}
}

func TestWaitHealthy_TimesOutIfNeverHealthy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	err := waitHealthy(srv.URL+"/health", 200*time.Millisecond)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
}
