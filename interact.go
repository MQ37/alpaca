package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

var stdin = bufio.NewReader(os.Stdin)

func readLine(prompt string) string {
	fmt.Print(prompt)
	line, _ := stdin.ReadString('\n')
	return strings.TrimSpace(line)
}

func promptDefault(prompt, def string) string {
	line := readLine(fmt.Sprintf("%s [%s]: ", prompt, def))
	if line == "" {
		return def
	}
	return line
}

func promptInt(prompt string, def int) int {
	line := promptDefault(prompt, strconv.Itoa(def))
	n, err := strconv.Atoi(line)
	if err != nil {
		return def
	}
	return n
}

func promptYesNo(prompt string, def bool) bool {
	d := "y/N"
	if def {
		d = "Y/n"
	}
	line := strings.ToLower(readLine(fmt.Sprintf("%s [%s]: ", prompt, d)))
	if line == "" {
		return def
	}
	return line == "y" || line == "yes"
}

func promptMode() string {
	fmt.Println("Select mode:")
	fmt.Println("  1) run    - interactive chat (llama-cli)")
	fmt.Println("  2) serve  - HTTP API server (llama-server)")
	for {
		switch readLine("> ") {
		case "1", "run":
			return "run"
		case "2", "serve":
			return "serve"
		}
	}
}

// promptModel lists models and returns the chosen index, reprompting on bad input.
func promptModel(models []Model) Model {
	fmt.Println("Available models:")
	for i, m := range models {
		fmt.Printf("  %2d) %s\n", i+1, m)
	}
	for {
		line := readLine("> ")
		if n, err := strconv.Atoi(line); err == nil && n >= 1 && n <= len(models) {
			return models[n-1]
		}
		fmt.Println("enter a number from the list")
	}
}

// matchModel resolves a -model flag value (index or substring) against the list.
func matchModel(models []Model, query string) (Model, bool) {
	if n, err := strconv.Atoi(query); err == nil && n >= 1 && n <= len(models) {
		return models[n-1], true
	}
	q := strings.ToLower(query)
	var hits []Model
	for _, m := range models {
		if strings.Contains(strings.ToLower(m.Repo+" "+m.Label), q) {
			hits = append(hits, m)
		}
	}
	if len(hits) == 1 {
		return hits[0], true
	}
	return Model{}, false
}
