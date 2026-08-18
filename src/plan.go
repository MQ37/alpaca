package main

import (
	"sort"
	"time"
)

// runningEntry is one currently-loaded model's memory footprint and last
// access time, as tracked by the swap supervisor.
type runningEntry struct {
	Bytes    int64
	LastUsed time.Time
}

// planEviction decides which currently-running models (oldest-used first)
// must be evicted to make room for a new model of newBytes under budget.
// Returns fits=false if newBytes alone exceeds budget even with every
// running model evicted — the caller must reject the request rather than
// spawn into a state that will immediately OOM.
func planEviction(running map[string]runningEntry, budget, newBytes int64) (evict []string, fits bool) {
	total := int64(0)
	ids := make([]string, 0, len(running))
	for id, e := range running {
		total += e.Bytes
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		return running[ids[i]].LastUsed.Before(running[ids[j]].LastUsed)
	})

	over := total + newBytes - budget
	if over <= 0 {
		return nil, true
	}

	for _, id := range ids {
		if over <= 0 {
			break
		}
		evict = append(evict, id)
		over -= running[id].Bytes
	}
	return evict, over <= 0
}
