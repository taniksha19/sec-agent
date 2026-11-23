// Package main is the API gateway in front of the FastAPI agent service.
// It checks API keys, rate-limits each key, logs every request, and forwards
// allowed requests to the agent. Standard library only.
package main

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	maxBodyBytes    = 16 << 10          // a question is at most 1000 characters
	upstreamTimeout = 120 * time.Second // one agent run; fails the request instead of hanging
	maxInFlight     = 2                 // concurrent agent runs per key; each run makes paid LLM calls
	minKeyLength    = 32                // long random keys make guessing hopeless
)

// routes the gateway forwards; everything else (including /docs) is a 404.
var routes = map[string]string{
	"/v1/ask": http.MethodPost,
}

// Gateway holds the keys, the per-key limits and the proxy to the agent.
type Gateway struct {
	keyHashes [][32]byte // SHA-256 of each valid key; the raw keys are not kept
	perMinute int
	proxy     *httputil.ReverseProxy
	log       *slog.Logger
	now       func() time.Time // replaced in tests

	mu    sync.Mutex
	usage map[string]*keyUsage // key id -> limits state; only valid keys get an entry
}

// keyUsage is one key's token bucket (up to perMinute tokens, refilled at perMinute per
// minute, one spent per request) and its count of requests still running.
type keyUsage struct {
	tokens   float64
	last     time.Time
	inFlight int
}

func NewGateway(keys []string, perMinute int, upstream *url.URL, log *slog.Logger) *Gateway {
	g := &Gateway{perMinute: perMinute, log: log, now: time.Now, usage: map[string]*keyUsage{}}
	for _, k := range keys {
		g.keyHashes = append(g.keyHashes, sha256.Sum256([]byte(k)))
	}
	g.proxy = &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(upstream)                // also sets Host to the upstream's, which the API allowlists
			r.SetXForwarded()                 // the real client address, not one the client claims
			r.Out.Header.Del("Authorization") // the agent never sees client keys
			r.Out.Header.Del("Forwarded")
			r.Out.Header.Del("X-Real-IP")
		},
		Transport: &http.Transport{
			DialContext:           (&net.Dialer{Timeout: 5 * time.Second}).DialContext,
			ResponseHeaderTimeout: upstreamTimeout,
		},
		ModifyResponse: func(r *http.Response) error {
			r.Header.Del("Server") // don't advertise the backend stack
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			var tooBig *http.MaxBytesError
			if errors.As(err, &tooBig) {
				http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
				return
			}
			log.Error("upstream failed", "error", err.Error())
			http.Error(w, "upstream unavailable", http.StatusBadGateway)
		},
	}
	return g
}

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := g.now()
	rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
	keyID := "-"
	requestID := newRequestID()
	rec.Header().Set("X-Request-ID", requestID)

	defer func() {
		g.log.Info("request",
			"request_id", requestID,
			"key_id", keyID,
			"client", r.RemoteAddr,
			"method", r.Method,
			"path", r.URL.Path,
			"status", rec.status,
			"latency_ms", g.now().Sub(start).Milliseconds(),
		)
	}()

	if r.URL.Path == "/healthz" && r.Method == http.MethodGet {
		rec.Write([]byte(`{"status":"ok"}`))
		return
	}
	method, ok := routes[r.URL.Path]
	if !ok {
		http.Error(rec, "not found", http.StatusNotFound)
		return
	}
	if r.Method != method {
		http.Error(rec, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	id, ok := g.authenticate(r)
	if !ok {
		rec.Header().Set("WWW-Authenticate", "Bearer")
		http.Error(rec, "invalid or missing API key", http.StatusUnauthorized)
		return
	}
	keyID = id

	if r.ContentLength > maxBodyBytes {
		http.Error(rec, "request body too large", http.StatusRequestEntityTooLarge)
		return
	}
	wait, ok := g.acquire(id)
	if !ok {
		if wait > 0 {
			rec.Header().Set("Retry-After", strconv.Itoa(int(math.Ceil(wait.Seconds()))))
		}
		http.Error(rec, "rate limit exceeded", http.StatusTooManyRequests)
		return
	}
	defer g.release(id)

	r.Body = http.MaxBytesReader(rec, r.Body, maxBodyBytes) // also caps bodies sent without a length
	r.Header.Set("X-Request-ID", requestID)
	g.proxy.ServeHTTP(rec, r)
}

// authenticate reads "Authorization: Bearer <key>" and returns a short id for logs.
// Every stored hash is compared in constant time, so timing doesn't reveal a near match.
func (g *Gateway) authenticate(r *http.Request) (string, bool) {
	scheme, key, found := strings.Cut(r.Header.Get("Authorization"), " ")
	if !found || !strings.EqualFold(scheme, "Bearer") || key == "" {
		return "", false
	}
	sum := sha256.Sum256([]byte(key))
	match := 0
	for _, h := range g.keyHashes {
		match |= subtle.ConstantTimeCompare(sum[:], h[:])
	}
	if match != 1 {
		return "", false
	}
	return hex.EncodeToString(sum[:4]), true // logs show this, never the key
}

// acquire admits one request for the key if it has a token and a free concurrency slot.
// When refused for lack of tokens it returns how long until the next one (0 if the key
// is only at its concurrency limit).
func (g *Gateway) acquire(keyID string) (time.Duration, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	now := g.now()
	capacity := float64(g.perMinute)
	u, ok := g.usage[keyID]
	if !ok {
		u = &keyUsage{tokens: capacity, last: now}
		g.usage[keyID] = u
	}
	perSecond := capacity / 60
	u.tokens = math.Min(capacity, u.tokens+now.Sub(u.last).Seconds()*perSecond)
	u.last = now
	if u.tokens < 1 {
		return time.Duration((1 - u.tokens) / perSecond * float64(time.Second)), false
	}
	if u.inFlight >= maxInFlight {
		return 0, false
	}
	u.tokens--
	u.inFlight++
	return 0, true
}

func (g *Gateway) release(keyID string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.usage[keyID].inFlight--
}

// statusRecorder remembers the status code so it can be logged.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (s *statusRecorder) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

func newRequestID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}
