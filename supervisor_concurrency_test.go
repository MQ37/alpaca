package main

import (
	"testing"
	"time"
)

// TestSupervisor_EnsureModel_AlreadyRunningLookupsDontBlockOnAnotherSpawn
// reproduces the bug where EnsureModel holds its lock across an entire
// slow spawn+health-check, serializing lookups for unrelated, already-
// running models behind it — defeating the point of running several
// models concurrently.
func TestSupervisor_EnsureModel_AlreadyRunningLookupsDontBlockOnAnotherSpawn(t *testing.T) {
	s := newSupervisor(1000<<30, 5*time.Second, func(id string, port int) (spawnedChild, error) {
		if id == "slow" {
			time.Sleep(300 * time.Millisecond)
		}
		return newFakeChild(true), nil
	})

	// pre-load "a" so it's already running
	if _, err := s.EnsureModel("a", 1<<30); err != nil {
		t.Fatalf("preload a: %v", err)
	}

	slowDone := make(chan struct{})
	go func() {
		s.EnsureModel("slow", 1<<30)
		close(slowDone)
	}()
	time.Sleep(20 * time.Millisecond) // let the slow spawn start and grab the lock

	start := time.Now()
	if _, err := s.EnsureModel("a", 1<<30); err != nil {
		t.Fatalf("lookup a: %v", err)
	}
	elapsed := time.Since(start)

	if elapsed > 100*time.Millisecond {
		t.Errorf("lookup of already-running model took %v, wanted <100ms (blocked behind unrelated slow spawn)", elapsed)
	}
	<-slowDone
}
