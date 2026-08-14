package main

import "testing"

func TestExtractModelMeta_UsesArchitecturePrefix(t *testing.T) {
	meta := ggufMetadata{
		"general.architecture":                 "muse-glimmer",
		"muse-glimmer.block_count":             uint32(52),
		"muse-glimmer.attention.head_count_kv": uint32(2),
		"muse-glimmer.attention.key_length":    uint32(128),
		"muse-glimmer.attention.value_length":  uint32(128),
	}

	m, err := extractModelMeta(meta)
	if err != nil {
		t.Fatalf("extractModelMeta: %v", err)
	}
	want := modelMeta{BlockCount: 52, HeadCountKV: 2, KeyLength: 128, ValueLength: 128}
	if m != want {
		t.Errorf("extractModelMeta = %+v, want %+v", m, want)
	}
}

func TestExtractModelMeta_MissingArchitectureErrors(t *testing.T) {
	_, err := extractModelMeta(ggufMetadata{})
	if err == nil {
		t.Fatal("expected error for missing general.architecture, got nil")
	}
}
