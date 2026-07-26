// alpaca is a small session launcher for llama.cpp binaries (llama-cli /
// llama-server). It discovers GGUF models already cached by the
// huggingface_hub client and drives llama-cli/llama-server with sane
// defaults (unified memory, GPU offload, context size, optional MTP
// speculative decoding).
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"syscall"
)

const (
	defaultCtx  = 4096
	defaultNGL  = 99
	defaultPort = 11212
	defaultMTPn = 2
)

// quirkOverrideKV maps repos with known-broken GGUF tokenizer metadata to
// the llama.cpp --override-kv fix needed for correct end-of-turn detection.
// unsloth/Laguna-S-2.1-GGUF: GGUF's eos/eot_token_id never got pointed at
// token 24 ("</assistant>", the model's real turn-end token), so the server
// never stops generating on its own. Verified via /tokenize + /props.
var quirkOverrideKV = map[string][]string{
	"unsloth/Laguna-S-2.1-GGUF": {"tokenizer.ggml.eot_token_id=int:24"},
}

func main() {
	if len(os.Args) < 2 {
		runInteractive("", nil)
		return
	}

	switch os.Args[1] {
	case "list":
		listModels()
		return
	case "run", "serve":
		runInteractive(os.Args[1], os.Args[2:])
		return
	case "-h", "--help", "help":
		usage()
		return
	}

	runInteractive("", os.Args[1:])
}

func usage() {
	fmt.Println(`alpaca - llama.cpp session manager

Usage:
  alpaca                          interactive: pick mode, model, context
  alpaca run    [flags] [-- extra llama-cli flags]
  alpaca serve  [flags] [-- extra llama-server flags]
  alpaca list                     print discovered models and exit

Flags:
  -model  <n|substr>   model index (from 'alpaca list') or name substring
  -ctx    <n>          context length (default 4096)
  -ngl    <n>          GPU layers to offload (default 99)
  -port   <n>          serve mode port (default 11212)
  -mtp                 enable MTP speculative decoding (draft-mtp)
  -mtp-n  <n>          MTP draft tokens (default 2)
  -cache  <dir>        override HF hub cache dir`)
}

func listModels() {
	models, err := discoverModels(defaultHubDir())
	must(err)
	if len(models) == 0 {
		fmt.Println("no GGUF models found in", defaultHubDir())
		return
	}
	for i, m := range models {
		fmt.Printf("%2d) %s\n", i+1, m)
	}
}

func runInteractive(mode string, args []string) {
	fs := flag.NewFlagSet("alpaca", flag.ExitOnError)
	modelQuery := fs.String("model", "", "model index or name substring")
	ctx := fs.Int("ctx", 0, "context length")
	ngl := fs.Int("ngl", defaultNGL, "GPU layers to offload")
	port := fs.Int("port", defaultPort, "serve mode port")
	mtp := fs.Bool("mtp", false, "enable MTP speculative decoding")
	mtpN := fs.Int("mtp-n", defaultMTPn, "MTP draft tokens")
	cacheDir := fs.String("cache", "", "override HF hub cache dir")
	fs.Parse(args)

	explicit := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { explicit[f.Name] = true })

	hubDir := *cacheDir
	if hubDir == "" {
		hubDir = defaultHubDir()
	}

	if mode == "" {
		mode = promptMode()
	}

	models, err := discoverModels(hubDir)
	must(err)
	if len(models) == 0 {
		fmt.Fprintln(os.Stderr, "no GGUF models found in", hubDir)
		os.Exit(1)
	}

	var model Model
	if *modelQuery != "" {
		m, ok := matchModel(models, *modelQuery)
		if !ok {
			fmt.Fprintf(os.Stderr, "no unique model matches %q, use 'alpaca list'\n", *modelQuery)
			os.Exit(1)
		}
		model = m
	} else {
		model = promptModel(models)
	}

	if !explicit["ctx"] && *ctx == 0 {
		*ctx = promptInt("Context length", defaultCtx)
	} else if *ctx == 0 {
		*ctx = defaultCtx
	}

	useMTP := *mtp
	if !explicit["mtp"] && model.MTP {
		useMTP = promptYesNo("MTP model detected, enable speculative decoding?", true)
	}
	if useMTP && !explicit["mtp-n"] {
		*mtpN = promptInt("MTP draft tokens", defaultMTPn)
	}

	binName := "llama-cli"
	if mode == "serve" {
		binName = "llama-server"
	}
	binPath, err := exec.LookPath(binName)
	must(err)

	cmdArgs := []string{binPath, "-m", model.Path, "-ngl", itoa(*ngl), "-c", itoa(*ctx)}
	if model.MMProj != "" {
		cmdArgs = append(cmdArgs, "--mmproj", model.MMProj)
	}
	if mode == "serve" {
		// -np 1: force single-slot serialized decoding. llama-server defaults
		// to -np -1 (auto, multiple concurrent slots), but this ROCm/HIP
		// build has a reproducible bug where a second concurrent request
		// while another slot is mid-generation comes back degenerate
		// (repeats a single garbage token forever). A coding agent only
		// ever needs one active generation at a time, so there's no
		// throughput cost to disabling slot concurrency here.
		cmdArgs = append(cmdArgs, "--port", itoa(*port), "--parallel", "1")
	}
	if useMTP {
		cmdArgs = append(cmdArgs, "--spec-type", "draft-mtp", "--spec-draft-n-max", itoa(*mtpN))
	}
	for _, kv := range quirkOverrideKV[model.Repo] {
		fmt.Printf("-> applying known fix: --override-kv %s\n", kv)
		cmdArgs = append(cmdArgs, "--override-kv", kv)
	}
	cmdArgs = append(cmdArgs, fs.Args()...)

	fmt.Printf("-> %s %s\n", binName, model)
	env := append(os.Environ(), "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1")
	must(syscall.Exec(binPath, cmdArgs, env))
}

func itoa(n int) string { return fmt.Sprintf("%d", n) }

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
