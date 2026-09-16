package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// fakeChild backs spawnedChild with a real local HTTP server, so
// waitHealthy's real HTTP polling exercises the actual code path without
// needing llama-server.
type fakeChild struct {
	srv     *httptest.Server
	stopped bool
	healthy bool
}

func newFakeChild(healthy bool) *fakeChild {
	fc := &fakeChild{healthy: healthy}
	fc.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fc.healthy {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	return fc
}

func (f *fakeChild) HealthURL() string   { return f.srv.URL + "/health" }
func (f *fakeChild) ProxyTarget() string { return f.srv.URL }
func (f *fakeChild) Stop(time.Duration) error {
	f.stopped = true
	f.srv.Close()
	return nil
}

func newTestSupervisor(budget int64, spawned map[string]*fakeChild) *supervisor {
	return newSupervisor(budget, 2*time.Second, func(id string, port int) (spawnedChild, error) {
		c := newFakeChild(true)
		spawned[id] = c
		return c, nil
	})
}

func TestSupervisor_EnsureModel_SpawnsWhenNotRunning(t *testing.T) {
	spawned := map[string]*fakeChild{}
	s := newTestSupervisor(100<<30, spawned)

	target, err := s.EnsureModel("gemma", 10<<30)
	if err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}
	if target != spawned["gemma"].ProxyTarget() {
		t.Errorf("target = %q, want %q", target, spawned["gemma"].ProxyTarget())
	}
}

func TestSupervisor_EnsureModel_ReusesAlreadyRunning(t *testing.T) {
	spawnCount := 0
	s := newSupervisor(100<<30, 2*time.Second, func(id string, port int) (spawnedChild, error) {
		spawnCount++
		return newFakeChild(true), nil
	})

	if _, err := s.EnsureModel("gemma", 10<<30); err != nil {
		t.Fatalf("first EnsureModel: %v", err)
	}
	if _, err := s.EnsureModel("gemma", 10<<30); err != nil {
		t.Fatalf("second EnsureModel: %v", err)
	}
	if spawnCount != 1 {
		t.Errorf("spawnCount = %d, want 1 (second call should reuse)", spawnCount)
	}
}

func TestSupervisor_EnsureModel_EvictsLRUWhenOverBudget(t *testing.T) {
	spawned := map[string]*fakeChild{}
	s := newTestSupervisor(50<<30, spawned)

	if _, err := s.EnsureModel("a", 30<<30); err != nil {
		t.Fatalf("spawn a: %v", err)
	}
	time.Sleep(5 * time.Millisecond) // ensure distinct LastUsed ordering
	if _, err := s.EnsureModel("b", 10<<30); err != nil {
		t.Fatalf("spawn b: %v", err)
	}
	// a(30)+b(10)=40 running, budget 50; requesting c(30) -> 70 total, over by 20
	// "a" is oldest (30GiB) -> evicting it covers the 20 needed
	if _, err := s.EnsureModel("c", 30<<30); err != nil {
		t.Fatalf("spawn c: %v", err)
	}

	if !spawned["a"].stopped {
		t.Error("expected 'a' (LRU) to be evicted/stopped")
	}
	if spawned["b"].stopped {
		t.Error("expected 'b' to remain running")
	}
}

func TestSupervisor_EnsureModel_RejectsIfNeverFits(t *testing.T) {
	spawned := map[string]*fakeChild{}
	s := newTestSupervisor(10<<30, spawned)

	_, err := s.EnsureModel("huge", 100<<30)
	if err == nil {
		t.Fatal("expected error, model alone exceeds entire budget")
	}
}

func TestSupervisor_EnsureModel_FailsHealthCheckCleansUp(t *testing.T) {
	var spawnedChildRef *fakeChild
	s := newSupervisor(100<<30, 300*time.Millisecond, func(id string, port int) (spawnedChild, error) {
		spawnedChildRef = newFakeChild(false) // never becomes healthy
		return spawnedChildRef, nil
	})

	_, err := s.EnsureModel("broken", 10<<30)
	if err == nil {
		t.Fatal("expected health-check failure error")
	}
	if !spawnedChildRef.stopped {
		t.Error("expected the unhealthy child to be stopped/cleaned up")
	}
}

func TestSupervisor_Evict_StopsAndForgetsRunningModel(t *testing.T) {
	spawned := map[string]*fakeChild{}
	s := newTestSupervisor(100<<30, spawned)

	if _, err := s.EnsureModel("gemma", 10<<30); err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}
	if !s.Evict("gemma") {
		t.Fatal("Evict returned false for a running model")
	}
	if !spawned["gemma"].stopped {
		t.Error("expected evicted child to be stopped")
	}

	// next EnsureModel must spawn a fresh one, not reuse the stopped child
	spawnCount := 0
	s.spawn = func(id string, port int) (spawnedChild, error) {
		spawnCount++
		c := newFakeChild(true)
		spawned[id] = c
		return c, nil
	}
	if _, err := s.EnsureModel("gemma", 10<<30); err != nil {
		t.Fatalf("EnsureModel after evict: %v", err)
	}
	if spawnCount != 1 {
		t.Errorf("spawnCount = %d, want 1 (evicted model must respawn)", spawnCount)
	}
}

func TestSupervisor_Evict_ReturnsFalseIfNotRunning(t *testing.T) {
	s := newTestSupervisor(100<<30, map[string]*fakeChild{})
	if s.Evict("never-loaded") {
		t.Error("expected false for a model that was never running")
	}
}

func TestSupervisor_Shutdown_StopsEverything(t *testing.T) {
	spawned := map[string]*fakeChild{}
	s := newTestSupervisor(100<<30, spawned)

	s.EnsureModel("a", 10<<30)
	s.EnsureModel("b", 10<<30)
	s.Shutdown(2 * time.Second)

	for id, c := range spawned {
		if !c.stopped {
			t.Errorf("expected %q to be stopped on shutdown", id)
		}
	}
}
