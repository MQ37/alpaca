package main

import (
	"testing"
	"time"
)

func TestPlanEviction_FitsWithoutEvicting(t *testing.T) {
	running := map[string]runningEntry{
		"gemma": {Bytes: 20 << 30, LastUsed: time.Now()},
	}
	evict, fits := planEviction(running, 100<<30, 30<<30)
	if len(evict) != 0 {
		t.Errorf("expected no eviction, got %v", evict)
	}
	if !fits {
		t.Error("expected fits=true")
	}
}

func TestPlanEviction_EvictsOldestFirstUntilItFits(t *testing.T) {
	now := time.Now()
	running := map[string]runningEntry{
		"oldest": {Bytes: 30 << 30, LastUsed: now.Add(-3 * time.Hour)},
		"middle": {Bytes: 30 << 30, LastUsed: now.Add(-2 * time.Hour)},
		"newest": {Bytes: 30 << 30, LastUsed: now.Add(-1 * time.Hour)},
	}
	// total running = 90GiB, budget = 100GiB, new model needs 40GiB
	// 90 + 40 = 130, over by 30 -> evicting "oldest" (30GiB) exactly covers it
	evict, fits := planEviction(running, 100<<30, 40<<30)
	if !fits {
		t.Fatal("expected fits=true")
	}
	if len(evict) != 1 || evict[0] != "oldest" {
		t.Errorf("evict = %v, want [oldest]", evict)
	}
}

func TestPlanEviction_EvictsMultipleOldestIfNeeded(t *testing.T) {
	now := time.Now()
	running := map[string]runningEntry{
		"oldest": {Bytes: 10 << 30, LastUsed: now.Add(-3 * time.Hour)},
		"middle": {Bytes: 10 << 30, LastUsed: now.Add(-2 * time.Hour)},
		"newest": {Bytes: 10 << 30, LastUsed: now.Add(-1 * time.Hour)},
	}
	// total = 30GiB, budget = 35GiB, new model needs 20GiB -> 30+20=50, over by 15
	// evict oldest (10) -> still over by 5 -> evict middle (10) -> now under
	evict, fits := planEviction(running, 35<<30, 20<<30)
	if !fits {
		t.Fatal("expected fits=true")
	}
	if len(evict) != 2 || evict[0] != "oldest" || evict[1] != "middle" {
		t.Errorf("evict = %v, want [oldest middle]", evict)
	}
}

func TestPlanEviction_DoesNotFitEvenAfterEvictingEverything(t *testing.T) {
	running := map[string]runningEntry{
		"a": {Bytes: 10 << 30, LastUsed: time.Now()},
	}
	// new model alone (200GiB) exceeds the entire 100GiB budget
	evict, fits := planEviction(running, 100<<30, 200<<30)
	if fits {
		t.Fatal("expected fits=false")
	}
	if len(evict) != 1 || evict[0] != "a" {
		t.Errorf("evict = %v, want [a] (everything evicted, still doesn't fit)", evict)
	}
}

func TestPlanEviction_EmptyRunningSet(t *testing.T) {
	evict, fits := planEviction(map[string]runningEntry{}, 100<<30, 10<<30)
	if len(evict) != 0 || !fits {
		t.Errorf("evict=%v fits=%v, want empty/true", evict, fits)
	}
}
