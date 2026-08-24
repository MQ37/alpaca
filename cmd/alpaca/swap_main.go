package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"
)

// registryEntry is one swap-managed model: its launchable Model plus the
// GGUF architecture metadata needed to estimate memory at a given context.
type registryEntry struct {
	Model Model
	Meta  modelMeta
}

// modelID is the swap-server-facing identifier for a discovered model,
// used in the "model" field of incoming requests.
func modelID(m Model) string { return m.Repo + "/" + m.Label }

// realChild adapts childProcess to the supervisor's spawnedChild interface
// for an actually-spawned llama-server.
type realChild struct {
	cp   *childProcess
	port int
}

func (r *realChild) HealthURL() string          { return fmt.Sprintf("http://127.0.0.1:%d/health", r.port) }
func (r *realChild) ProxyTarget() string        { return fmt.Sprintf("http://127.0.0.1:%d", r.port) }
func (r *realChild) Stop(d time.Duration) error { return r.cp.Stop(d) }

func runSwap(args []string) {
	fs := flag.NewFlagSet("swap", flag.ExitOnError)
	listen := fs.String("listen", ":8090", "address to listen on")
	ctx := fs.Int("ctx", defaultCtx, "context length for every swap-managed model")
	ngl := fs.Int("ngl", defaultNGL, "GPU layers to offload")
	reasoningBudget := fs.Int("reasoning-budget", 0, "default --reasoning-budget for every swap-managed model (0 = omit flag, llama-server default -1/unrestricted); per-model override via -settings")
	memMarginGB := fs.Int64("mem-margin-gb", defaultGPUMemMargin>>30, "GB of GPU memory reserved for the OS, excluded from the swap budget")
	memBudgetGB := fs.Int64("mem-budget-gb", 0, "override auto-detected GPU memory budget (GB); 0 = auto-detect from /sys/class/drm")
	healthTimeout := fs.Duration("health-timeout", 120*time.Second, "how long to wait for a spawned model to become healthy")
	cacheDir := fs.String("cache", "", "override HF hub cache dir")
	settingsPath := fs.String("settings", "", "JSON file of per-model overrides, e.g. {\"<repo>/<label>\": {\"ctx\": N}}")
	fs.Parse(args)

	settings, err := loadSettings(*settingsPath)
	must(err)

	hubDir := *cacheDir
	if hubDir == "" {
		hubDir = defaultHubDir()
	}
	models, err := discoverModels(hubDir)
	must(err)
	if len(models) == 0 {
		fmt.Fprintln(os.Stderr, "no GGUF models found in", hubDir)
		os.Exit(1)
	}

	registry := make(map[string]registryEntry, len(models))
	for _, m := range models {
		meta, metaErr := loadModelMeta(m.Path)
		if metaErr != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s (failed to read GGUF metadata: %v)\n", modelID(m), metaErr)
			continue
		}
		registry[modelID(m)] = registryEntry{Model: m, Meta: meta}
	}
	if len(registry) == 0 {
		fmt.Fprintln(os.Stderr, "no models with readable GGUF metadata found, nothing to serve")
		os.Exit(1)
	}

	binPath, err := exec.LookPath("llama-server")
	must(err)

	budget := *memBudgetGB << 30
	if budget == 0 {
		var berr error
		budget, berr = detectGPUBudget(*memMarginGB << 30)
		must(berr)
	}
	fmt.Printf("swap budget: %.1f GiB (%d models discovered)\n", float64(budget)/(1<<30), len(registry))

	spawn := func(id string, port int) (spawnedChild, error) {
		entry, ok := registry[id]
		if !ok {
			return nil, fmt.Errorf("model %q not in registry", id)
		}
		llamaArgs := buildLlamaArgs(llamaArgsConfig{
			ModelPath:       entry.Model.Path,
			MMProj:          entry.Model.MMProj,
			NGL:             *ngl,
			Ctx:             ctxFor(settings, id, *ctx),
			Serve:           true,
			Port:            port,
			Parallel:        1,
			MTP:             entry.Model.MTP,
			MTPN:            defaultMTPn,
			ReasoningBudget: reasoningBudgetFor(settings, id, *reasoningBudget),
			OverrideKV:      quirkOverrideKV[entry.Model.Repo],
		})
		cmd := exec.Command(binPath, llamaArgs...)
		cmd.Env = append(os.Environ(), "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cp := &childProcess{}
		if err := cp.Start(cmd); err != nil {
			return nil, err
		}
		fmt.Printf("-> spawning %s on port %d\n", id, port)
		return &realChild{cp: cp, port: port}, nil
	}

	sup := newSupervisor(budget, *healthTimeout, spawn)

	estimate := func(id string) (int64, error) {
		entry, ok := registry[id]
		if !ok {
			return 0, errUnknownModel
		}
		return estimateMemoryBytes(entry.Model.SizeBytes, entry.Meta, ctxFor(settings, id, *ctx)), nil
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("shutting down, stopping loaded models...")
		sup.Shutdown(defaultStopGrace)
		os.Exit(0)
	}()

	fmt.Printf("alpaca swap listening on %s\n", *listen)
	must(http.ListenAndServe(*listen, newSwapHandler(sup, estimate)))
}
