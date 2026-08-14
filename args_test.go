package main

import (
	"reflect"
	"testing"
)

func TestBuildLlamaArgs_RunMode_Minimal(t *testing.T) {
	got := buildLlamaArgs(llamaArgsConfig{
		ModelPath: "/models/foo.gguf",
		NGL:       99,
		Ctx:       4096,
	})
	want := []string{"-m", "/models/foo.gguf", "-ngl", "99", "-c", "4096"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestBuildLlamaArgs_ServeMode_AddsPortAndParallel(t *testing.T) {
	got := buildLlamaArgs(llamaArgsConfig{
		ModelPath: "/models/foo.gguf",
		NGL:       99,
		Ctx:       4096,
		Serve:     true,
		Port:      11212,
		Parallel:  1,
	})
	want := []string{
		"-m", "/models/foo.gguf", "-ngl", "99", "-c", "4096",
		"--port", "11212", "--parallel", "1",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestBuildLlamaArgs_FullOptions_OrderMatchesOriginalInlineLogic(t *testing.T) {
	got := buildLlamaArgs(llamaArgsConfig{
		ModelPath:  "/models/foo.gguf",
		MMProj:     "/models/mmproj-foo.gguf",
		NGL:        99,
		Ctx:        8192,
		Serve:      true,
		Port:       11212,
		Parallel:   1,
		MTP:        true,
		MTPN:       3,
		OverrideKV: []string{"tokenizer.ggml.eot_token_id=int:24"},
		Extra:      []string{"--verbose"},
	})
	want := []string{
		"-m", "/models/foo.gguf", "-ngl", "99", "-c", "8192",
		"--mmproj", "/models/mmproj-foo.gguf",
		"--port", "11212", "--parallel", "1",
		"--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
		"--override-kv", "tokenizer.ggml.eot_token_id=int:24",
		"--verbose",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestBuildLlamaArgs_RunMode_OmitsPortAndParallel(t *testing.T) {
	got := buildLlamaArgs(llamaArgsConfig{
		ModelPath: "/models/foo.gguf",
		NGL:       99,
		Ctx:       4096,
		Serve:     false,
		Port:      11212, // must be ignored when Serve is false
	})
	for _, arg := range got {
		if arg == "--port" {
			t.Fatalf("run mode must not include --port, got %v", got)
		}
	}
}
