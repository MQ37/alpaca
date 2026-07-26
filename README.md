<p align="center">
  <i>Point it at your HF hub cache, pick a model, run.</i>
</p>

<p align="center">
  <img alt="Go 1.25" src="https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white">
  <img alt="dependencies" src="https://img.shields.io/badge/deps-stdlib%20only-3fb950">
  <img alt="single binary" src="https://img.shields.io/badge/build-single%20static%20binary-238636">
</p>

---

**alpaca** is a small session launcher for `llama.cpp` (`llama-cli` /
`llama-server`). It discovers GGUF models already cached by the
`huggingface_hub` client, lets you pick one interactively (or via flags), and
execs `llama-cli`/`llama-server` with sane defaults — GPU offload, unified
memory, context size, optional MTP speculative decoding.

```bash
go build -o alpaca .
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
- **Two modes** — `run` (`llama-cli`, interactive chat) and `serve`
  (`llama-server`, HTTP API on `-port`, default `11212`).
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

## 📦 Dependencies

Zero third-party dependencies — standard library only (`flag`, `os/exec`,
`syscall`, `path/filepath`, `regexp`). `go.mod` declares no `require`s.

---

## 🧭 Design

- **Single static binary**, no config file — flags + env vars only.
- **Reads the HF hub cache layout directly** — no `huggingface_hub` Python
  dependency at runtime, just the on-disk convention it writes.
- **Suckless-ish** — 4 files, ~350 lines total, one job.
