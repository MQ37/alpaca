package main

import "strconv"

// llamaArgsConfig is everything needed to build a llama-cli/llama-server
// argv, independent of how the resulting process gets launched
// (syscall.Exec self-replace for run/serve, or os/exec.Command supervised
// for swap mode).
type llamaArgsConfig struct {
	ModelPath string
	MMProj    string
	NGL       int
	Ctx       int
	Serve     bool // adds --port/--parallel; false for run mode (llama-cli)
	Port      int
	Parallel  int
	MTP       bool
	MTPN      int
	// ReasoningBudget sets --reasoning-budget (token cap on thinking) when
	// non-zero; 0 omits the flag (llama-server's own default: -1/unrestricted).
	// Some models (e.g. certain Gemma GGUF imports) run away generating
	// <unused*> filler under llama-server's default unrestricted budget —
	// see docs/known-issues.md.
	ReasoningBudget int
	OverrideKV      []string // quirk fixes for known-broken GGUF metadata
	Extra           []string // pass-through CLI flags after everything else
}

// buildLlamaArgs assembles the llama-cli/llama-server argv (excluding the
// binary path itself, which the caller supplies to exec/Command).
func buildLlamaArgs(cfg llamaArgsConfig) []string {
	args := []string{"-m", cfg.ModelPath, "-ngl", strconv.Itoa(cfg.NGL), "-c", strconv.Itoa(cfg.Ctx)}
	if cfg.MMProj != "" {
		args = append(args, "--mmproj", cfg.MMProj)
	}
	if cfg.Serve {
		args = append(args, "--port", strconv.Itoa(cfg.Port), "--parallel", strconv.Itoa(cfg.Parallel))
	}
	if cfg.MTP {
		args = append(args, "--spec-type", "draft-mtp", "--spec-draft-n-max", strconv.Itoa(cfg.MTPN))
	}
	if cfg.ReasoningBudget != 0 {
		args = append(args, "--reasoning-budget", strconv.Itoa(cfg.ReasoningBudget))
	}
	for _, kv := range cfg.OverrideKV {
		args = append(args, "--override-kv", kv)
	}
	args = append(args, cfg.Extra...)
	return args
}
