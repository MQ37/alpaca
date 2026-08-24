package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadSettings_MissingFile_ReturnsEmpty(t *testing.T) {
	settings, err := loadSettings(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(settings) != 0 {
		t.Fatalf("settings = %v, want empty", settings)
	}
}

func TestLoadSettings_EmptyPath_ReturnsEmpty(t *testing.T) {
	settings, err := loadSettings("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(settings) != 0 {
		t.Fatalf("settings = %v, want empty", settings)
	}
}

func TestLoadSettings_ParsesCtxOverrides(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	body := `{
		"unsloth/Laguna-S-2.1-GGUF/UD-Q4_K_XL/Laguna-S-2.1-UD-Q4_K_XL": {"ctx": 32768},
		"unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL": {"ctx": 16384}
	}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	settings, err := loadSettings(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := settings["unsloth/Laguna-S-2.1-GGUF/UD-Q4_K_XL/Laguna-S-2.1-UD-Q4_K_XL"].Ctx; got != 32768 {
		t.Errorf("Laguna ctx = %d, want 32768", got)
	}
	if got := settings["unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL"].Ctx; got != 16384 {
		t.Errorf("gemma ctx = %d, want 16384", got)
	}
}

func TestLoadSettings_InvalidJSON_Errors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(path, []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadSettings(path); err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
}

func TestCtxFor_UsesOverrideWhenPositive(t *testing.T) {
	settings := map[string]modelSettings{"model-a": {Ctx: 32768}}
	if got := ctxFor(settings, "model-a", 4096); got != 32768 {
		t.Errorf("ctxFor = %d, want 32768", got)
	}
}

func TestCtxFor_FallsBackWhenAbsent(t *testing.T) {
	settings := map[string]modelSettings{}
	if got := ctxFor(settings, "model-a", 4096); got != 4096 {
		t.Errorf("ctxFor = %d, want fallback 4096", got)
	}
}

func TestCtxFor_FallsBackWhenZero(t *testing.T) {
	settings := map[string]modelSettings{"model-a": {Ctx: 0}}
	if got := ctxFor(settings, "model-a", 4096); got != 4096 {
		t.Errorf("ctxFor = %d, want fallback 4096", got)
	}
}

func TestLoadSettings_ParsesReasoningBudget(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	body := `{"some/model": {"ctx": 262144, "reasoning_budget": 1024}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	settings, err := loadSettings(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := settings["some/model"].ReasoningBudget; got != 1024 {
		t.Errorf("ReasoningBudget = %d, want 1024", got)
	}
}

func TestReasoningBudgetFor_UsesOverrideWhenNonZero(t *testing.T) {
	settings := map[string]modelSettings{"model-a": {ReasoningBudget: 1024}}
	if got := reasoningBudgetFor(settings, "model-a", 0); got != 1024 {
		t.Errorf("reasoningBudgetFor = %d, want 1024", got)
	}
}

func TestReasoningBudgetFor_FallsBackWhenAbsentOrZero(t *testing.T) {
	settings := map[string]modelSettings{"model-a": {}}
	if got := reasoningBudgetFor(settings, "model-a", -1); got != -1 {
		t.Errorf("reasoningBudgetFor = %d, want fallback -1", got)
	}
	if got := reasoningBudgetFor(map[string]modelSettings{}, "model-b", 5); got != 5 {
		t.Errorf("reasoningBudgetFor = %d, want fallback 5", got)
	}
}

func TestLoadSettings_ParsesExtra(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	body := `{"some/model": {"extra": ["--cache-ram", "0"]}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	settings, err := loadSettings(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []string{"--cache-ram", "0"}
	if got := settings["some/model"].Extra; !reflect.DeepEqual(got, want) {
		t.Errorf("Extra = %v, want %v", got, want)
	}
}

func TestExtraFor_ReturnsSettingsExtraOrNilWhenAbsent(t *testing.T) {
	settings := map[string]modelSettings{"model-a": {Extra: []string{"--cache-ram", "0"}}}
	if got := extraFor(settings, "model-a"); !reflect.DeepEqual(got, []string{"--cache-ram", "0"}) {
		t.Errorf("extraFor = %v, want [--cache-ram 0]", got)
	}
	if got := extraFor(settings, "model-b"); len(got) != 0 {
		t.Errorf("extraFor for absent model = %v, want empty", got)
	}
}
