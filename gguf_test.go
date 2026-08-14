package main

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// buildGGUF assembles a minimal valid GGUF byte stream with the given
// scalar uint32 metadata key/value pairs. No tensors, matching real files
// (we only ever read the metadata header).
func buildGGUF(t *testing.T, kv map[string]uint32) []byte {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3)) // version
	binary.Write(&buf, binary.LittleEndian, uint64(0)) // tensor_count
	binary.Write(&buf, binary.LittleEndian, uint64(len(kv)))

	// deterministic order for test readability
	keys := make([]string, 0, len(kv))
	for k := range kv {
		keys = append(keys, k)
	}
	for _, k := range keys {
		binary.Write(&buf, binary.LittleEndian, uint64(len(k)))
		buf.WriteString(k)
		binary.Write(&buf, binary.LittleEndian, uint32(4)) // type: uint32
		binary.Write(&buf, binary.LittleEndian, kv[k])
	}
	return buf.Bytes()
}

func TestParseGGUFMetadata_ReadsUint32Fields(t *testing.T) {
	data := buildGGUF(t, map[string]uint32{
		"muse-glimmer.block_count":             52,
		"muse-glimmer.attention.head_count_kv": 2,
		"muse-glimmer.attention.key_length":    128,
	})

	meta, err := parseGGUFMetadata(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("parseGGUFMetadata: %v", err)
	}

	if got := meta.uint32("muse-glimmer.block_count"); got != 52 {
		t.Errorf("block_count = %d, want 52", got)
	}
	if got := meta.uint32("muse-glimmer.attention.head_count_kv"); got != 2 {
		t.Errorf("head_count_kv = %d, want 2", got)
	}
}

func TestParseGGUFMetadata_SkipsArrayValuesAndKeepsReadingAfter(t *testing.T) {
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(0))
	binary.Write(&buf, binary.LittleEndian, uint64(2)) // 2 kv pairs

	// key 1: an array of 3 uint32s (e.g. tokenizer.ggml.tokens-like field)
	key1 := "some.array"
	binary.Write(&buf, binary.LittleEndian, uint64(len(key1)))
	buf.WriteString(key1)
	binary.Write(&buf, binary.LittleEndian, uint32(ggufArray))
	binary.Write(&buf, binary.LittleEndian, uint32(ggufUint32)) // element type
	binary.Write(&buf, binary.LittleEndian, uint64(3))          // length
	binary.Write(&buf, binary.LittleEndian, uint32(1))
	binary.Write(&buf, binary.LittleEndian, uint32(2))
	binary.Write(&buf, binary.LittleEndian, uint32(3))

	// key 2: a plain scalar that must still parse correctly after the array
	key2 := "after.array"
	binary.Write(&buf, binary.LittleEndian, uint64(len(key2)))
	buf.WriteString(key2)
	binary.Write(&buf, binary.LittleEndian, uint32(ggufUint32))
	binary.Write(&buf, binary.LittleEndian, uint32(99))

	meta, err := parseGGUFMetadata(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("parseGGUFMetadata: %v", err)
	}
	if got := meta.uint32("after.array"); got != 99 {
		t.Errorf("after.array = %d, want 99 (stream desynced after array skip)", got)
	}
}

func TestParseGGUFMetadata_RejectsBadMagic(t *testing.T) {
	data := []byte("NOPE0000")
	_, err := parseGGUFMetadata(bytes.NewReader(data))
	if err == nil {
		t.Fatal("expected error for bad magic, got nil")
	}
}
