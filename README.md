<p align="center">
  <i>Point it at your HF hub cache, pick a model, run.</i>
</p>

<p align="center">
  <img alt="Go 1.25" src="https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white">
  <img alt="dependencies" src="https://img.shields.io/badge/deps-stdlib%20only-3fb950">
  <img alt="single binary" src="https://img.shields.io/badge/build-single%20static%20binary-238636">
</p>

---

**alpaca** is a session launcher and memory-aware swap server for
`llama.cpp` (`llama-cli` / `llama-server`). It discovers GGUF models already
cached by the `huggingface_hub` client, lets you pick one interactively (or
via flags), and execs `llama-cli`/`llama-server` with sane defaults — GPU
offload, unified memory, context size, optional MTP speculative decoding.
`alpaca swap` runs a long-lived OpenAI-compatible proxy that loads/evicts
models on demand, keeping several loaded at once if they fit the GPU memory
budget.

Source lives in `cmd/alpaca/`; `go.mod` is at the repo root (same layout as
most Go CLIs — one `cmd/<binary>` package per binary).

```bash
go install github.com/MQ37/alpaca/cmd/alpaca@latest   # installs $GOBIN/alpaca
```

Or build from a clone:

```bash
git clone https://github.com/MQ37/alpaca && cd alpaca
go build -o alpaca ./cmd/alpaca
./alpaca              # interactive: pick mode, model, context
```

---

## ✨ Features

- **Model discovery** — walks `~/.cache/huggingface/hub` (or `$HF_HOME`,
  `$HUGGINGFACE_HUB_CACHE`, `-cache <dir>`) for `models--*/snapshots/*/**/*.gguf`.
  Groups split shards (`-00001-of-00005`) and sidecar `mmproj-*.gguf` files
  into one launchable entry each; dedupes repos that re-resolve to the same
  blob under a new snapshot hash.
- **Interactive by default** — no args prompts for mode (run/serve), model
  (numbered list), and context length. Any flag you pass explicitly skips its
  prompt.
- **Three modes** — `run` (`llama-cli`, interactive chat), `serve`
  (`llama-server`, HTTP API on `-port`, default `11212`), and `swap`
  (memory-aware multi-model swap server, see below).
- **MTP speculative decoding** — auto-detected from repo/label name
  (`mtp` substring); prompts to enable `--spec-type draft-mtp` with
  `-mtp-n` draft tokens (default 2).
- **`alpaca list`** — print discovered models and exit, no prompts.
- **`-model <n|substr>`** — pick by list index or a unique name substring,
  skipping the picker entirely.
- **Execs, doesn't fork** — replaces itself (`syscall.Exec`) with
  `llama-cli`/`llama-server`, so signals/ctrl-c/exit codes pass straight
  through; sets `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`.

---

## 🚀 Usage

```bash
alpaca                          # interactive: pick mode, model, context
alpaca run    [flags] [-- extra llama-cli flags]
alpaca serve  [flags] [-- extra llama-server flags]
alpaca swap   [flags]           # memory-aware multi-model swap server
alpaca list                     # print discovered models and exit
```

Flags:

| Flag | Default | What |
|---|---|---|
| `-model <n\|substr>` | prompt | model index (from `alpaca list`) or name substring |
| `-ctx <n>` | 4096 | context length |
| `-ngl <n>` | 99 | GPU layers to offload |
| `-port <n>` | 11212 | serve mode port |
| `-mtp` | off | enable MTP speculative decoding |
| `-mtp-n <n>` | 2 | MTP draft tokens |
| `-cache <dir>` | HF hub default | override HF hub cache dir |

---

## 🔀 `alpaca swap`

A long-lived OpenAI/llama.cpp-compatible HTTP proxy. Every request's
`model` field selects a discovered GGUF; if it isn't already running,
`alpaca swap` estimates its memory footprint (on-disk weight size + a
KV-cache calculation from GGUF architecture metadata + a fixed compute
overhead), evicts the least-recently-used loaded model(s) only if needed
to fit under the GPU memory budget, spawns it, waits for `/health`, then
reverse-proxies the request through — including streaming responses.

Unlike a single-model swap, several models stay loaded concurrently
whenever their combined estimate fits the budget, instead of always
evicting the current one.

```bash
alpaca swap -listen :8090
```

| Flag | Default | What |
|---|---|---|
| `-listen <addr>` | `:8090` | address to listen on |
| `-ctx <n>` | 4096 | context length applied to every swap-managed model |
| `-ngl <n>` | 99 | GPU layers to offload |
| `-mem-budget-gb <n>` | auto | override the auto-detected GPU memory budget |
| `-mem-margin-gb <n>` | 10 | GB reserved for the OS, excluded from the budget |
| `-health-timeout <d>` | 120s | how long to wait for a spawned model to become healthy |
| `-cache <dir>` | HF hub default | override HF hub cache dir |
| `-settings <path>` | none | JSON file of per-model overrides (see below) |
| `-reasoning-budget <n>` | 0 (omit flag) | default `--reasoning-budget` for every model; per-model override via `-settings` |

**Per-model settings**: `-ctx`/`-reasoning-budget` set one value for every
model, which is wrong when models differ — a 15GB model and a 68GB model
don't have the same memory headroom, and some GGUF imports need a bounded
reasoning budget while others don't. `-settings` points at a JSON file
keyed by the same `<repo>/<label>` model ID:

```json
{
  "unsloth/Laguna-S-2.1-GGUF/UD-Q4_K_XL/Laguna-S-2.1-UD-Q4_K_XL": {"ctx": 65536},
  "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL": {"ctx": 131072},
  "unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL": {"ctx": 262144, "reasoning_budget": 1024}
}
```

`ctx: 0`/`reasoning_budget: 0`/absent all fall back to the `-ctx`/
`-reasoning-budget` flag defaults.

Pick `ctx` values that actually fit: KV cache scales linearly with `ctx`,
and a large weight model at a huge context can exceed the whole GPU budget
by itself (`fits=false` — the request then always 503s, never mind
eviction). Check `swap budget: N GiB` at startup against your models'
sizes first.

`reasoning_budget` caps `llama-server`'s own thinking-token budget
(default: `-1`, unrestricted). Some Gemma GGUF imports run away generating
`<unused*>` filler tokens under an unrestricted budget — see
[llama.cpp#21321](https://github.com/ggml-org/llama.cpp/issues/21321); a
bounded budget (e.g. `1024`) fixes it.

**Model name in requests**: the `model` field must be the exact
`<repo>/<label>` string from `alpaca list` (e.g.
`unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL`), not a
short name — anything else gets a `404 unknown model`:

```bash
curl http://127.0.0.1:8090/v1/chat/completions -d '{
  "model": "unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL",
  "messages": [{"role": "user", "content": "hi"}]
}'
```

The budget auto-detects from
`/sys/class/drm/card*/device/mem_info_{vram,gtt}_total` (AMD unified
memory) minus the margin; override with `-mem-budget-gb` on other GPUs.

---

## 📦 Dependencies

Zero third-party dependencies — standard library only (`flag`, `os/exec`,
`syscall`, `path/filepath`, `regexp`). `go.mod` declares no `require`s.

---

## 🧭 Design

- **Layout**: `go.mod` + all Go source under `cmd/alpaca/` at the repo root
  (standard Go `cmd/<binary>` layout, same as `go install`-friendly CLIs).
  Build with `go build -o alpaca ./cmd/alpaca`.
- **Single static binary**, no config file — flags + env vars only.
- **Reads the HF hub cache layout directly** — no `huggingface_hub` Python
  dependency at runtime, just the on-disk convention it writes.
- **Suckless-ish** — one file per concern (GGUF parsing, memory estimate,
  eviction planning, process supervision, HTTP proxy), no framework, no
  external process manager.
