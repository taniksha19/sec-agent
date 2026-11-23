package main

import (
	"bytes"
	"crypto/sha256"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

const goodKey = "test-key-0123456789abcdef0123456789"

// setup starts a fake agent that records what it received, and a gateway in front of it.
func setup(t *testing.T, perMinute int) (*Gateway, *http.Request) {
	t.Helper()
	seen := httptest.NewRequest("GET", "/", nil) // replaced by the last request the agent saw
	agent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*seen = *r
		w.Header().Set("Server", "uvicorn")
		w.Write([]byte(`{"answer":"ok"}`))
	}))
	t.Cleanup(agent.Close)
	upstream, _ := url.Parse(agent.URL)
	log := slog.New(slog.NewJSONHandler(io.Discard, nil))
	return NewGateway([]string{goodKey, "other-key"}, perMinute, upstream, log), seen
}

func ask(g *Gateway, key string, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest("POST", "/v1/ask", strings.NewReader(body))
	if key != "" {
		r.Header.Set("Authorization", "Bearer "+key)
	}
	w := httptest.NewRecorder()
	g.ServeHTTP(w, r)
	return w
}

func TestGoodKeyIsForwardedWithoutTheKey(t *testing.T) {
	g, seen := setup(t, 10)
	w := ask(g, goodKey, `{"question":"q"}`)
	if w.Code != 200 || w.Body.String() != `{"answer":"ok"}` {
		t.Fatalf("got %d %q", w.Code, w.Body.String())
	}
	if seen.Header.Get("Authorization") != "" {
		t.Error("the agent must not receive the client's key")
	}
	if seen.Header.Get("X-Request-ID") == "" || w.Header().Get("X-Request-ID") == "" {
		t.Error("request id missing")
	}
	if w.Header().Get("Server") != "" {
		t.Error("backend Server header should be hidden")
	}
}

func TestSpoofedForwardingHeadersAreReplaced(t *testing.T) {
	g, seen := setup(t, 10)
	r := httptest.NewRequest("POST", "/v1/ask", strings.NewReader("{}"))
	r.Header.Set("Authorization", "bearer "+goodKey) // scheme is case-insensitive
	r.Header.Set("Forwarded", "for=1.2.3.4")
	r.Header.Set("X-Real-IP", "1.2.3.4")
	r.Header.Set("X-Forwarded-For", "1.2.3.4")
	w := httptest.NewRecorder()
	g.ServeHTTP(w, r)
	if w.Code != 200 {
		t.Fatalf("got %d", w.Code)
	}
	if seen.Header.Get("Forwarded") != "" || seen.Header.Get("X-Real-IP") != "" {
		t.Error("client-set forwarding headers must be dropped")
	}
	if xff := seen.Header.Get("X-Forwarded-For"); xff != "192.0.2.1" { // httptest's client address
		t.Errorf("X-Forwarded-For = %q, want the real client address", xff)
	}
}

func TestConcurrentRequestsPerKeyAreCapped(t *testing.T) {
	g, _ := setup(t, 10)
	g.acquire(keyIDFor(goodKey))
	g.acquire(keyIDFor(goodKey)) // two runs in flight
	w := ask(g, goodKey, `{}`)
	if w.Code != http.StatusTooManyRequests || w.Header().Get("Retry-After") != "" {
		t.Fatalf("got %d retry-after %q, want 429 without Retry-After", w.Code, w.Header().Get("Retry-After"))
	}
	g.release(keyIDFor(goodKey))
	if w := ask(g, goodKey, `{}`); w.Code != 200 {
		t.Errorf("after one finished: got %d", w.Code)
	}
}

func keyIDFor(key string) string {
	r := httptest.NewRequest("POST", "/v1/ask", nil)
	r.Header.Set("Authorization", "Bearer "+key)
	id, _ := (&Gateway{keyHashes: [][32]byte{sha256.Sum256([]byte(key))}}).authenticate(r)
	return id
}

func TestBadOrMissingKeyIsRejected(t *testing.T) {
	g, _ := setup(t, 10)
	for _, key := range []string{"", "wrong", goodKey + "x", "Bearer " + goodKey} {
		if w := ask(g, key, `{}`); w.Code != http.StatusUnauthorized {
			t.Errorf("key %q: got %d, want 401", key, w.Code)
		}
	}
	r := httptest.NewRequest("POST", "/v1/ask", nil)
	r.Header.Set("Authorization", "Basic "+goodKey)
	w := httptest.NewRecorder()
	g.ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("non-Bearer scheme: got %d", w.Code)
	}
}

func TestRateLimitPerKeyAndRefill(t *testing.T) {
	g, _ := setup(t, 3)
	clock := time.Unix(1_000_000, 0)
	g.now = func() time.Time { return clock }

	for i := range 3 {
		if w := ask(g, goodKey, `{}`); w.Code != 200 {
			t.Fatalf("request %d: got %d", i, w.Code)
		}
	}
	w := ask(g, goodKey, `{}`)
	if w.Code != http.StatusTooManyRequests || w.Header().Get("Retry-After") != "20" {
		t.Fatalf("got %d retry-after %q, want 429 and 20", w.Code, w.Header().Get("Retry-After"))
	}
	if w := ask(g, "other-key", `{}`); w.Code != 200 {
		t.Errorf("another key has its own limit: got %d", w.Code)
	}
	clock = clock.Add(20 * time.Second) // 3 per minute refills one token every 20 s
	if w := ask(g, goodKey, `{}`); w.Code != 200 {
		t.Errorf("after refill: got %d", w.Code)
	}
}

func TestOnlyKnownRoutes(t *testing.T) {
	g, _ := setup(t, 10)
	cases := []struct {
		method, path string
		want         int
	}{
		{"GET", "/healthz", 200}, // no key needed
		{"GET", "/docs", 404},
		{"GET", "/openapi.json", 404},
		{"GET", "/v1/ask", 405},
		{"POST", "//v1/ask", 404},
		{"POST", "/v1/ask/", 404},
		{"POST", "/V1/ask", 404},
	}
	for _, c := range cases {
		r := httptest.NewRequest(c.method, c.path, nil)
		r.Header.Set("Authorization", "Bearer "+goodKey)
		w := httptest.NewRecorder()
		g.ServeHTTP(w, r)
		if w.Code != c.want {
			t.Errorf("%s %s: got %d, want %d", c.method, c.path, w.Code, c.want)
		}
	}
}

func TestLargeBodyIsRejected(t *testing.T) {
	g, _ := setup(t, 10)
	big := bytes.Repeat([]byte("a"), maxBodyBytes+1)
	if w := ask(g, goodKey, string(big)); w.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("got %d, want 413", w.Code)
	}
}

func TestAgentDownIs502(t *testing.T) {
	upstream, _ := url.Parse("http://127.0.0.1:1") // nothing listens here
	g := NewGateway([]string{goodKey}, 10, upstream, slog.New(slog.NewJSONHandler(io.Discard, nil)))
	if w := ask(g, goodKey, `{}`); w.Code != http.StatusBadGateway {
		t.Errorf("got %d, want 502", w.Code)
	}
}

func TestSplitKeys(t *testing.T) {
	got := splitKeys(" a, ,b ,")
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("got %q", got)
	}
}

func TestCheckKeys(t *testing.T) {
	if checkKeys(nil) == nil {
		t.Error("no keys must fail")
	}
	err := checkKeys([]string{goodKey, "short-secret"})
	if err == nil || strings.Contains(err.Error(), "short-secret") {
		t.Errorf("short key must fail without echoing the key: %v", err)
	}
	if checkKeys([]string{goodKey}) != nil {
		t.Error("a long key is fine")
	}
}
