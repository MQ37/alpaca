package main

import "testing"

func TestKVCacheBytes_MatchesLlamaCppFormula(t *testing.T) {
	// KV cache = n_layer * ctx * n_head_kv * (key_len + value_len) * bytes_per_elem
	// 10 * 1000 * 4 * (8+8) * 2 = 1,280,000
	got := kvCacheBytes(modelMeta{
		BlockCount:  10,
		HeadCountKV: 4,
		KeyLength:   8,
		ValueLength: 8,
	}, 1000)
	want := int64(1_280_000)
	if got != want {
		t.Errorf("kvCacheBytes = %d, want %d", got, want)
	}
}

func TestEstimateMemoryBytes_SumsWeightsKVAndOverhead(t *testing.T) {
	meta := modelMeta{BlockCount: 10, HeadCountKV: 4, KeyLength: 8, ValueLength: 8}
	weightBytes := int64(5_000_000_000)
	ctx := 1000

	got := estimateMemoryBytes(weightBytes, meta, ctx)
	want := weightBytes + kvCacheBytes(meta, ctx) + computeBufferOverhead
	if got != want {
		t.Errorf("estimateMemoryBytes = %d, want %d", got, want)
	}
}

func TestEstimateMemoryBytes_ZeroHeadCountKVFallsBackToNoKVCost(t *testing.T) {
	// a model missing/zeroed GQA metadata shouldn't panic or divide by zero;
	// it degrades to weights + overhead only (a known-conservative underestimate).
	meta := modelMeta{}
	got := estimateMemoryBytes(1000, meta, 4096)
	want := int64(1000) + computeBufferOverhead
	if got != want {
		t.Errorf("estimateMemoryBytes = %d, want %d", got, want)
	}
}
