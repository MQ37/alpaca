package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Model is one launchable GGUF found in the HF hub cache.
type Model struct {
	Repo      string // "unsloth/Laguna-S-2.1-GGUF"
	Label     string // quant/subdir + base filename, e.g. "UD-Q4_K_M/Laguna-S-2.1-UD-Q4_K_M"
	Path      string // path passed to -m (first shard if split)
	MMProj    string // sibling mmproj-*.gguf, or ""
	SizeBytes int64  // summed across shards
	Parts     int
	MTP       bool // repo/label name suggests MTP speculative-decoding model
}

var splitRe = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})$`)

func defaultHubDir() string {
	if v := os.Getenv("HF_HOME"); v != "" {
		return filepath.Join(v, "hub")
	}
	if v := os.Getenv("HUGGINGFACE_HUB_CACHE"); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

func repoFromDirName(name string) string {
	rest := strings.TrimPrefix(name, "models--")
	parts := strings.SplitN(rest, "--", 2)
	if len(parts) == 2 {
		return parts[0] + "/" + parts[1]
	}
	return rest
}

// discoverModels walks hubDir/models--*/snapshots/*/**/*.gguf and groups
// split shards and mmproj sidecars into single Model entries.
func discoverModels(hubDir string) ([]Model, error) {
	entries, err := os.ReadDir(hubDir)
	if err != nil {
		return nil, err
	}

	var models []Model
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "models--") {
			continue
		}
		repo := repoFromDirName(e.Name())
		snapDir := filepath.Join(hubDir, e.Name(), "snapshots")
		snaps, err := os.ReadDir(snapDir)
		if err != nil {
			continue
		}
		for _, snap := range snaps {
			if !snap.IsDir() {
				continue
			}
			models = append(models, scanSnapshot(repo, filepath.Join(snapDir, snap.Name()))...)
		}
	}

	models = dedupeByContent(models)

	sort.Slice(models, func(i, j int) bool {
		if models[i].Repo != models[j].Repo {
			return models[i].Repo < models[j].Repo
		}
		return models[i].Label < models[j].Label
	})
	return models, nil
}

// dedupeByContent drops entries whose primary shard resolves to the same
// blob as one already seen. HF re-resolves a repo ref to a new snapshot hash
// even when the underlying blobs are unchanged (e.g. a metadata-only commit),
// which otherwise lists the identical model twice.
func dedupeByContent(models []Model) []Model {
	seen := make(map[string]bool, len(models))
	out := make([]Model, 0, len(models))
	for _, m := range models {
		real, err := filepath.EvalSymlinks(m.Path)
		if err != nil {
			real = m.Path
		}
		if seen[real] {
			continue
		}
		seen[real] = true
		out = append(out, m)
	}
	return out
}

// shard groups one split-file family (or a lone single-file model).
type shard struct {
	dir   string
	base  string // filename without split suffix and without .gguf
	parts map[int]string
	size  int64
}

func scanSnapshot(repo, root string) []Model {
	var gguf []string
	filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(d.Name()), ".gguf") {
			gguf = append(gguf, path)
		}
		return nil
	})

	mmproj := map[string]string{} // dir -> mmproj path
	shards := map[string]*shard{} // dir+base -> shard

	for _, path := range gguf {
		dir := filepath.Dir(path)
		name := strings.TrimSuffix(filepath.Base(path), ".gguf")

		if strings.HasPrefix(strings.ToLower(name), "mmproj") {
			mmproj[dir] = path
			continue
		}

		base := name
		part := 1
		if m := splitRe.FindStringSubmatch(name); m != nil {
			base = m[1]
			fmt.Sscanf(m[2], "%d", &part)
		}

		key := dir + "|" + base
		s := shards[key]
		if s == nil {
			s = &shard{dir: dir, base: base, parts: map[int]string{}}
			shards[key] = s
		}
		s.parts[part] = path
		if info, err := os.Stat(path); err == nil {
			s.size += info.Size()
		}
	}

	var out []Model
	for _, s := range shards {
		firstPart := s.parts[1]
		if firstPart == "" {
			// no part numbered 1 found (shouldn't happen); pick any
			for _, p := range s.parts {
				firstPart = p
				break
			}
		}
		rel, _ := filepath.Rel(root, s.dir)
		label := s.base
		if rel != "." {
			label = rel + "/" + s.base
		}
		m := Model{
			Repo:      repo,
			Label:     label,
			Path:      firstPart,
			MMProj:    mmproj[s.dir],
			SizeBytes: s.size,
			Parts:     len(s.parts),
		}
		lower := strings.ToLower(repo + " " + label)
		m.MTP = strings.Contains(lower, "mtp")
		out = append(out, m)
	}
	return out
}

func humanSize(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

func (m Model) String() string {
	extra := ""
	if m.Parts > 1 {
		extra = fmt.Sprintf(" (%d parts)", m.Parts)
	}
	if m.MMProj != "" {
		extra += " +mmproj"
	}
	if m.MTP {
		extra += " [MTP]"
	}
	return fmt.Sprintf("%s :: %s  [%s]%s", m.Repo, m.Label, humanSize(m.SizeBytes), extra)
}
