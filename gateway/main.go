package main

import (
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	keys := splitKeys(os.Getenv("GATEWAY_API_KEYS"))
	if err := checkKeys(keys); err != nil {
		log.Error(err.Error())
		os.Exit(1)
	}
	upstream, err := url.Parse(envOr("UPSTREAM_URL", "http://api:8000"))
	if err != nil {
		log.Error("bad UPSTREAM_URL", "error", err.Error())
		os.Exit(1)
	}
	perMinute, err := strconv.Atoi(envOr("RATE_LIMIT_PER_MINUTE", "10"))
	if err != nil || perMinute < 1 {
		log.Error("RATE_LIMIT_PER_MINUTE must be a positive integer")
		os.Exit(1)
	}

	server := &http.Server{
		Addr:              ":8080",
		Handler:           NewGateway(keys, perMinute, upstream, log),
		ReadHeaderTimeout: 5 * time.Second, // slow clients can't hold connections open
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      upstreamTimeout + 5*time.Second,
		IdleTimeout:       60 * time.Second,
	}
	log.Info("gateway listening", "addr", server.Addr, "keys", len(keys), "per_minute", perMinute)
	if err := server.ListenAndServe(); err != nil {
		log.Error("server stopped", "error", err.Error())
		os.Exit(1)
	}
}

// splitKeys parses a comma-separated key list, ignoring blanks.
func splitKeys(s string) []string {
	var keys []string
	for _, k := range strings.Split(s, ",") {
		if k = strings.TrimSpace(k); k != "" {
			keys = append(keys, k)
		}
	}
	return keys
}

// checkKeys refuses to start with no keys or with a short one (which could be guessed).
func checkKeys(keys []string) error {
	if len(keys) == 0 {
		return errors.New("GATEWAY_API_KEYS is empty; refusing to start without keys")
	}
	for i, k := range keys {
		if len(k) < minKeyLength {
			// Says which key by position, never the key itself.
			return fmt.Errorf("GATEWAY_API_KEYS key %d is shorter than %d characters", i+1, minKeyLength)
		}
	}
	return nil
}

func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
