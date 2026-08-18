package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// modelSettings is a per-model override of swap defaults, keyed by the
// same "<repo>/<label>" id used in request bodies and the registry.
type modelSettings struct {
	Ctx int `json:"ctx"` // 0 = fall back to the -ctx flag default
}

// loadSettings reads a JSON file of {"<model-id>": {"ctx": N}, ...}. An
// empty path or a missing file is not an error — every model then uses the
// -ctx flag default.
func loadSettings(path string) (map[string]modelSettings, error) {
	if path == "" {
		return map[string]modelSettings{}, nil
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return map[string]modelSettings{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading settings file %s: %w", path, err)
	}
	var settings map[string]modelSettings
	if err := json.Unmarshal(data, &settings); err != nil {
		return nil, fmt.Errorf("parsing settings file %s: %w", path, err)
	}
	return settings, nil
}

// ctxFor resolves the effective context length for a model: its settings
// override when present and positive, else fallback (the -ctx flag).
func ctxFor(settings map[string]modelSettings, id string, fallback int) int {
	if s, ok := settings[id]; ok && s.Ctx > 0 {
		return s.Ctx
	}
	return fallback
}
