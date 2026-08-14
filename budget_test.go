package main

import (
	"os"
	"path/filepath"
	"testing"
)

func writeSysfsFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadSysfsUint64_ParsesTrimmedDecimal(t *testing.T) {
	dir := t.TempDir()
	path := writeSysfsFile(t, dir, "vram_total", "1073741824\n")

	got, err := readSysfsUint64(path)
	if err != nil {
		t.Fatalf("readSysfsUint64: %v", err)
	}
	if got != 1073741824 {
		t.Errorf("got %d, want 1073741824", got)
	}
}

func TestGPUBudgetBytes_SumsPoolsMinusMargin(t *testing.T) {
	dir := t.TempDir()
	vram := writeSysfsFile(t, dir, "vram_total", "1073741824") // 1 GiB
	gtt := writeSysfsFile(t, dir, "gtt_total", "118111600640") // ~110 GiB

	got, err := gpuBudgetBytes(vram, gtt, 10<<30) // 10 GiB margin
	if err != nil {
		t.Fatalf("gpuBudgetBytes: %v", err)
	}
	want := int64(1073741824+118111600640) - int64(10<<30)
	if got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestGPUBudgetBytes_MarginExceedingTotalFloorsAtZero(t *testing.T) {
	dir := t.TempDir()
	vram := writeSysfsFile(t, dir, "vram_total", "100")
	gtt := writeSysfsFile(t, dir, "gtt_total", "100")

	got, err := gpuBudgetBytes(vram, gtt, 1<<40) // absurdly large margin
	if err != nil {
		t.Fatalf("gpuBudgetBytes: %v", err)
	}
	if got != 0 {
		t.Errorf("got %d, want 0 (floored)", got)
	}
}
