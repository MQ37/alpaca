package main

// modelMeta is the subset of GGUF architecture metadata needed to estimate
// a model's runtime memory footprint.
type modelMeta struct {
	BlockCount  uint32 // n_layer
	HeadCountKV uint32 // n_head_kv (post-GQA head count)
	KeyLength   uint32 // per-head key dim
	ValueLength uint32 // per-head value dim
}

// kvCacheBytesPerElem assumes llama.cpp's default fp16 KV cache. Quantized
// KV cache (q8_0/q4_0) would need this to be a parameter; not needed yet
// since alpaca doesn't expose --cache-type-k/v.
const kvCacheBytesPerElem = 2

// computeBufferOverhead is a fixed safety margin for llama.cpp's compute
// graph buffers, which this estimate doesn't model precisely. Chosen
// conservatively; recalibrate against real rocm-smi readings if models
// still OOM near the budget edge.
const computeBufferOverhead = 2 << 30 // 2 GiB

// kvCacheBytes computes llama.cpp's KV cache size for ctx tokens:
// n_layer * ctx * n_head_kv * (key_len + value_len) * bytes_per_elem.
func kvCacheBytes(m modelMeta, ctx int) int64 {
	return int64(m.BlockCount) * int64(ctx) * int64(m.HeadCountKV) *
		int64(m.KeyLength+m.ValueLength) * kvCacheBytesPerElem
}

// estimateMemoryBytes is the total memory alpaca budgets for one loaded
// model: on-disk weight bytes (already ~1:1 with loaded size for
// pre-quantized GGUF) plus KV cache for the requested context plus a flat
// compute-buffer overhead.
func estimateMemoryBytes(weightBytes int64, meta modelMeta, ctx int) int64 {
	return weightBytes + kvCacheBytes(meta, ctx) + computeBufferOverhead
}
