package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// defaultGPUMemMargin is reserved for the OS/other processes and left out
// of the swap budget. Bastion's own headroom (123GB RAM, ~110GB GTT after
// the amdgpu.gttsize fix) leaves ~12GB for the OS at this margin.
const defaultGPUMemMargin = 10 << 30 // 10 GiB

func readSysfsUint64(path string) (uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}

// gpuBudgetBytes sums the VRAM and GTT pools reported at vramPath/gttPath
// and reserves marginBytes for the OS. Floors at 0 rather than going
// negative if margin exceeds total (a misconfigured margin shouldn't panic
// downstream arithmetic).
func gpuBudgetBytes(vramPath, gttPath string, marginBytes int64) (int64, error) {
	vram, err := readSysfsUint64(vramPath)
	if err != nil {
		return 0, fmt.Errorf("read vram total: %w", err)
	}
	gtt, err := readSysfsUint64(gttPath)
	if err != nil {
		return 0, fmt.Errorf("read gtt total: %w", err)
	}
	total := int64(vram) + int64(gtt) - marginBytes
	if total < 0 {
		return 0, nil
	}
	return total, nil
}

// detectGPUBudget finds the first amdgpu card's memory pools under sysfs
// and returns the swap budget after reserving marginBytes for the OS.
func detectGPUBudget(marginBytes int64) (int64, error) {
	vramMatches, err := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_total")
	if err != nil || len(vramMatches) == 0 {
		return 0, fmt.Errorf("no amdgpu vram_total found under /sys/class/drm")
	}
	gttPath := filepath.Join(filepath.Dir(vramMatches[0]), "mem_info_gtt_total")
	return gpuBudgetBytes(vramMatches[0], gttPath, marginBytes)
}
