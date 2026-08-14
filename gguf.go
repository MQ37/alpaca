package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// ggufValueType mirrors the GGUF spec's metadata value type tags.
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
const (
	ggufUint8   = 0
	ggufInt8    = 1
	ggufUint16  = 2
	ggufInt16   = 3
	ggufUint32  = 4
	ggufInt32   = 5
	ggufFloat32 = 6
	ggufBool    = 7
	ggufString  = 8
	ggufArray   = 9
	ggufUint64  = 10
	ggufInt64   = 11
	ggufFloat64 = 12
)

// ggufMetadata holds the scalar metadata key/value pairs read from a GGUF
// file header. Only scalars are kept; array values are skipped (nothing
// alpaca needs lives in an array field).
type ggufMetadata map[string]any

func (m ggufMetadata) uint32(key string) uint32 {
	v, _ := m[key].(uint32)
	return v
}

// parseGGUFMetadata reads a GGUF file's header and metadata key/value
// section only — it never reads tensor info or tensor data, so it's cheap
// even against a 100GB blob.
func parseGGUFMetadata(r io.Reader) (ggufMetadata, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return nil, fmt.Errorf("not a GGUF file: bad magic %q", magic)
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}

	var tensorCount, kvCount uint64
	if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("read tensor_count: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &kvCount); err != nil {
		return nil, fmt.Errorf("read metadata_kv_count: %w", err)
	}

	meta := make(ggufMetadata, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, err := readGGUFString(r)
		if err != nil {
			return nil, fmt.Errorf("read key %d: %w", i, err)
		}
		var valType uint32
		if err := binary.Read(r, binary.LittleEndian, &valType); err != nil {
			return nil, fmt.Errorf("read value type for %q: %w", key, err)
		}
		val, err := readGGUFValue(r, valType)
		if err != nil {
			return nil, fmt.Errorf("read value for %q: %w", key, err)
		}
		meta[key] = val
	}
	return meta, nil
}

func readGGUFString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readGGUFValue reads one scalar value of valType, or fully consumes (and
// discards) an array value so the stream stays aligned for the next key.
func readGGUFValue(r io.Reader, valType uint32) (any, error) {
	switch valType {
	case ggufUint8, ggufInt8, ggufBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufUint16, ggufInt16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufUint32, ggufInt32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufUint64, ggufInt64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufString:
		return readGGUFString(r)
	case ggufArray:
		return nil, skipGGUFArray(r)
	default:
		return nil, fmt.Errorf("unknown GGUF value type %d", valType)
	}
}

// extractModelMeta pulls the architecture-prefixed fields needed for
// memory estimation out of raw GGUF metadata.
func extractModelMeta(meta ggufMetadata) (modelMeta, error) {
	arch, ok := meta["general.architecture"].(string)
	if !ok || arch == "" {
		return modelMeta{}, fmt.Errorf("missing general.architecture key")
	}
	return modelMeta{
		BlockCount:  meta.uint32(arch + ".block_count"),
		HeadCountKV: meta.uint32(arch + ".attention.head_count_kv"),
		KeyLength:   meta.uint32(arch + ".attention.key_length"),
		ValueLength: meta.uint32(arch + ".attention.value_length"),
	}, nil
}

// loadModelMeta reads a GGUF file's metadata header and extracts modelMeta.
func loadModelMeta(path string) (modelMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return modelMeta{}, err
	}
	defer f.Close()

	meta, err := parseGGUFMetadata(f)
	if err != nil {
		return modelMeta{}, err
	}
	return extractModelMeta(meta)
}

func skipGGUFArray(r io.Reader) error {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return err
	}
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return err
	}
	for i := uint64(0); i < length; i++ {
		if _, err := readGGUFValue(r, elemType); err != nil {
			return err
		}
	}
	return nil
}
