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

## 3. `unsloth/gemma-4-26B-A4B-it-GGUF` generates unbounded `<unused*>` filler tokens

**Symptom:** at large context (`ctx=262144`, the model's own advertised max),
even a trivial prompt ("hello bro") returns nothing but `<unused49><unused49>...`
repeated until `max_tokens`. Confirmed the same architecture family
(`gemma4`), same specific model, and same symptom (`<unused24>` there) is a
known upstream bug, not something wrong with this GGUF specifically or with
context size: [ggml-org/llama.cpp#21321](https://github.com/ggml-org/llama.cpp/issues/21321).

**Root cause:** `llama-server`'s own reasoning/thinking-budget tracking
defaults to `-1` (unrestricted, printed internally as `2147483647`). Gemma 4's
chat template activates a "thinking" phase per response; with no cap, the
model can run away generating reasoning filler that degenerates into
`<unused*>` placeholder tokens instead of ever reaching real content. Not a
context-size, rope-scaling, or SWA-cache bug — reproduced this exact failure
even on a 5-word prompt (position ~10), and reproduced it going away at the
same `ctx=262144` with the budget capped, including on a long realistic
tool-heavy multi-turn prompt.

**Ruled out first**: rope-scaling metadata (none embedded — this arch's
`context_length=262144` is native RoPE with hybrid sliding-window attention,
no YaRN to misconfigure) and `--swa-full` (correctness flag for hybrid-SWA
architectures — OOMs at this size, needs one 50GB contiguous KV allocation
even with the GPU otherwise idle).

**Workaround (what alpaca does)**: per-model `reasoning_budget` in
`-settings` (see § alpaca swap above) passes `--reasoning-budget N` at spawn
time. `alpaca`'s own bastion deployment caps this model at `8192` — verified
clean at full `ctx=262144` on both a trivial prompt and an 800/1500-token
LHC explanation (the same question from the upstream bug report) with zero
`<unused*>` tokens in either the reasoning or content fields.

**Second, separate finding — corruption persists across unrelated requests**:
a client that sends an OpenAI-style `reasoning_effort` body field (poisson's
Ollama-compatible provider does, for its `/effort` setting) can still trigger
the same `<unused*>` runaway despite `--reasoning-budget` being set — this
field is read by `llama-server` per-request and isn't gated by the CLI flag.
Worse: once one generation degenerates, the corruption doesn't stay confined
to that one request — every *subsequent* request on the process, including
ones with no `reasoning_effort` field at all and completely unrelated
content, comes back corrupted too, until the process is restarted. Root
cause: `llama-server`'s prompt-cache / KV-cache-reuse (`--cache-ram`, default
8192 MiB — logged as `selected slot by LCP similarity, f_sim_best=...`)
reuses cached KV state across requests that share a prefix; once a
generation corrupts that cached state, reuse spreads the corruption to
everything after it on that slot.

**Workaround (what alpaca does)**: per-model `extra` in `-settings` passes
raw CLI flags at spawn time; this model's entry adds `["--cache-ram", "0"]`
to fully disable prompt-cache reuse, so a bad generation can never poison a
later, unrelated request. Verified: 5 back-to-back requests each including
`reasoning_effort: "medium"` all came back clean with this flag set (all 5
failed/degenerated without it in the same conditions before this fix).
