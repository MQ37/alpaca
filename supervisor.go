package main

import (
	"fmt"
	"sync"
	"time"
)

// spawnedChild is the handle a spawn function returns for one launched
// model process. Implemented by the real llama-server child in production
// and by a fake HTTP server in tests.
type spawnedChild interface {
	HealthURL() string
	ProxyTarget() string
	Stop(gracePeriod time.Duration) error
}

const defaultStopGrace = 10 * time.Second

// startPort is the first port handed out to spawned children; each new
// model gets the next one, avoiding collisions between concurrently
// running models.
const startPort = 11300

type runningModelState struct {
	entry runningEntry
	child spawnedChild
}

// pendingSpawn lets concurrent EnsureModel calls for the same not-yet-
// running id wait on the one spawn already in flight instead of racing
// to spawn it twice.
type pendingSpawn struct {
	done chan struct{}
}

// supervisor is alpaca swap's core: decides which model to keep loaded,
// evicts LRU models under memory pressure, and spawns new ones on demand.
// Memory estimation (GGUF parsing, KV cache math) lives elsewhere — callers
// pass in an already-computed estimatedBytes so this stays a pure lifecycle
// manager.
//
// mu only ever guards map bookkeeping (running/pending/nextPort), never the
// slow parts (spawning a process, health-check polling, stopping a child):
// those run unlocked so a cold spawn for one model can't stall lookups of
// other already-running models.
type supervisor struct {
	mu            sync.Mutex
	budget        int64
	healthTimeout time.Duration
	spawn         func(id string, port int) (spawnedChild, error)
	nextPort      int
	running       map[string]*runningModelState
	pending       map[string]*pendingSpawn
}

func newSupervisor(budget int64, healthTimeout time.Duration, spawn func(id string, port int) (spawnedChild, error)) *supervisor {
	return &supervisor{
		budget:        budget,
		healthTimeout: healthTimeout,
		spawn:         spawn,
		nextPort:      startPort,
		running:       make(map[string]*runningModelState),
		pending:       make(map[string]*pendingSpawn),
	}
}

// EnsureModel returns a proxy target URL for id, spawning it (and evicting
// LRU models if needed to fit estimatedBytes under budget) if not already
// running. Concurrent calls for different already-running ids never block
// on each other or on a third id's in-flight spawn.
func (s *supervisor) EnsureModel(id string, estimatedBytes int64) (string, error) {
	for {
		s.mu.Lock()
		if rm, ok := s.running[id]; ok {
			rm.entry.LastUsed = time.Now()
			target := rm.child.ProxyTarget()
			s.mu.Unlock()
			return target, nil
		}
		if p, ok := s.pending[id]; ok {
			s.mu.Unlock()
			<-p.done
			continue // re-check running/pending: the in-flight spawn just finished (or failed)
		}

		snapshot := make(map[string]runningEntry, len(s.running))
		for k, v := range s.running {
			snapshot[k] = v.entry
		}
		evictIDs, fits := planEviction(snapshot, s.budget, estimatedBytes)
		if !fits {
			s.mu.Unlock()
			return "", fmt.Errorf("model %q needs %d bytes, exceeds budget %d bytes even after evicting every other loaded model", id, estimatedBytes, s.budget)
		}

		toStop := make([]spawnedChild, 0, len(evictIDs))
		for _, evID := range evictIDs {
			toStop = append(toStop, s.running[evID].child)
			delete(s.running, evID)
		}

		port := s.nextPort
		s.nextPort++
		p := &pendingSpawn{done: make(chan struct{})}
		s.pending[id] = p
		s.mu.Unlock()

		for _, child := range toStop {
			child.Stop(defaultStopGrace)
		}

		target, spawnErr := s.doSpawn(id, port, estimatedBytes)

		s.mu.Lock()
		delete(s.pending, id)
		s.mu.Unlock()
		close(p.done)

		return target, spawnErr
	}
}

// doSpawn runs the slow, unlocked part of loading a model: spawn the
// process, wait for it to become healthy, and register it as running on
// success.
func (s *supervisor) doSpawn(id string, port int, estimatedBytes int64) (string, error) {
	child, err := s.spawn(id, port)
	if err != nil {
		return "", fmt.Errorf("spawn %q: %w", id, err)
	}

	if err := waitHealthy(child.HealthURL(), s.healthTimeout); err != nil {
		child.Stop(defaultStopGrace)
		return "", fmt.Errorf("model %q failed health check: %w", id, err)
	}

	s.mu.Lock()
	s.running[id] = &runningModelState{
		entry: runningEntry{Bytes: estimatedBytes, LastUsed: time.Now()},
		child: child,
	}
	s.mu.Unlock()
	return child.ProxyTarget(), nil
}

// Shutdown stops every running child, e.g. on SIGTERM/SIGINT.
func (s *supervisor) Shutdown(gracePeriod time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, rm := range s.running {
		rm.child.Stop(gracePeriod)
		delete(s.running, id)
	}
}
