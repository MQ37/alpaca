# Known upstream issues alpaca works around

## 1. Concurrent multi-slot decode corruption on integrated HIP GPUs (gfx1151 / Strix Halo)

**Symptom:** with `llama-server` running its default `-np -1` (auto, resolves to
multiple slots), any request that arrives while another slot is still
generating comes back corrupted — degenerated into a single repeated garbage
token forever (never stops), or in more severe cases returns another
request's answer verbatim, or a "chimeric" fusion of two different requests'
output.

**Reproduced locally**: fired a trivial fresh prompt (`"say ok"`) via
`/completion` with `return_tokens:true` while a real, unrelated, long-running
generation was active in another slot. Result: `tokens: [14, 14, 14, ...]`
repeated forever, empty `content`, `stop: true` only once `n_predict` was hit.
Confirmed the server process itself stayed responsive (`/health` kept
returning 200, `/slots` answered instantly) — only the *new* concurrent
request's output was corrupted, not the whole server.

**Upstream report**: [ggml-org/llama.cpp#25992](https://github.com/ggml-org/llama.cpp/issues/25992)
— "server -np 4 --kv-unified returns other requests' responses verbatim on
integrated HIP GPU (gfx1151)", filed against the same hardware class
(Strix Halo / gfx1151, 128GB unified memory, ROCm). Bisected to a specific
commit (`c7d8722`, "ggml-cuda: restore prop.integrated on HIP builds",
PR #24233) — root cause is specific to the HIP backend's
`prop.integrated == 1` path (integrated/APU GPUs only; discrete AMD GPUs are
unaffected), triggered by multiple slots + unified KV cache + concurrent
requests. Open, unfixed as of this writing.

**Workaround (what alpaca does)**: `alpaca serve` always launches
`llama-server` with `--parallel 1`, forcing single-slot serialized decoding.
This has no real cost for a coding-agent workload (one active conversation at
a time) and completely avoids the concurrent-batch code path where the
corruption happens — a queued second request just waits its turn instead of
coming back garbled.

## 2. `unsloth/Laguna-S-2.1-GGUF` never stops generating (broken GGUF tokenizer metadata)

**Symptom:** every response runs until `n_predict`/context limit instead of
stopping naturally — `llama-server` logs `special_eos_id is not in
special_eog_ids` / `special_eot_id is not in special_eog_ids` at load time.

**Root cause:** the model's actual end-of-turn marker is a real, single vocab
token — `</assistant>` (token id **24**, confirmed via `/tokenize`) — but the
GGUF conversion never pointed `tokenizer.ggml.eot_token_id` at it, so
`llama.cpp` has no way to know generation should stop there. This is a
tokenizer/metadata bug in the model conversion, not a code bug in llama.cpp
or alpaca; the same GGUF misbehaves identically whether driven by
`llama-cli`, `llama-server`, or its built-in web UI (all three share the
exact same vocab-loading and `/v1/chat/completions` code path in this
llama.cpp build — `llama-cli` literally embeds `llama-server` and talks to it
over HTTP, see `tools/cli/cli-server.h`).

**Workaround (what alpaca does)**: both `alpaca run` and `alpaca serve` (see
`quirkOverrideKV` in `main.go`) automatically add
`--override-kv tokenizer.ggml.eot_token_id=int:24` when launching this
repo, which fixes end-of-turn detection at model-load time without touching
the cached `.gguf` file.
