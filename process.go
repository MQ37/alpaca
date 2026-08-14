package main

import (
	"fmt"
	"net/http"
	"os/exec"
	"syscall"
	"time"
)

// childProcess wraps one supervised llama-server child. Unlike alpaca's
// run/serve modes (which syscall.Exec-replace themselves), swap mode stays
// alive and must be able to start, health-check, and stop several of these
// over its lifetime.
type childProcess struct {
	cmd *exec.Cmd
}

// Start launches cmd and returns once the process has been forked (does not
// wait for it to become ready — callers use waitHealthy for that).
func (c *childProcess) Start(cmd *exec.Cmd) error {
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start: %w", err)
	}
	c.cmd = cmd
	return nil
}

// Stop sends SIGTERM and waits up to gracePeriod for a clean exit, then
// SIGKILLs if the process is still alive.
func (c *childProcess) Stop(gracePeriod time.Duration) error {
	if c.cmd == nil || c.cmd.Process == nil {
		return nil
	}

	done := make(chan error, 1)
	go func() { done <- c.cmd.Wait() }()

	_ = c.cmd.Process.Signal(syscall.SIGTERM)

	select {
	case <-done:
		return nil
	case <-time.After(gracePeriod):
		_ = c.cmd.Process.Kill()
		<-done // reap after force-kill
		return nil
	}
}

// waitHealthy polls url until it returns HTTP 200 or timeout elapses.
func waitHealthy(url string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 2 * time.Second}

	for {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for %s to become healthy", url)
		}
		time.Sleep(200 * time.Millisecond)
	}
}
