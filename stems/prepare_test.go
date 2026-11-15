package stems

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

type assetTransport func(*http.Request) (*http.Response, error)

func (f assetTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func testAsset(body []byte, format string) asset {
	sum := sha256.Sum256(body)
	return asset{URL: "https://example.invalid/runtime", SHA256: hex.EncodeToString(sum[:]), Format: format, Target: "model.onnx"}
}
func assetClient(body []byte, calls *atomic.Int32) *http.Client {
	return &http.Client{Transport: assetTransport(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		return &http.Response{StatusCode: 200, Body: io.NopCloser(bytes.NewReader(body)), Header: make(http.Header)}, nil
	})}
}

func TestPrepareCacheOfflineAndIntegrity(t *testing.T) {
	ctx := context.Background()
	body := []byte("model")
	a := testAsset(body, "file")
	opts := PrepareOptions{CacheDir: t.TempDir()}
	var calls atomic.Int32
	client := assetClient(body, &calls)
	dir, err := prepare(ctx, opts, []asset{a}, client)
	if err != nil {
		t.Fatal(err)
	}
	opts.Offline = true
	again, err := prepare(ctx, opts, []asset{a}, client)
	if err != nil || again != dir {
		t.Fatalf("cache: %s %v", again, err)
	}
	if calls.Load() != 1 {
		t.Fatal("cache accessed network")
	}
	if err := os.WriteFile(filepath.Join(dir, "model.onnx"), []byte("damaged"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := prepare(ctx, opts, []asset{a}, client); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatalf("corrupt cache: %v", err)
	}
	if calls.Load() != 1 {
		t.Fatal("offline mode accessed network")
	}
	opts.CacheDir = t.TempDir()
	if _, err := prepare(ctx, opts, []asset{a}, client); err == nil {
		t.Fatal("accepted missing offline assets")
	}
	if calls.Load() != 1 {
		t.Fatal("offline mode accessed network")
	}
}

func TestPrepareRejectsBadDownloadAndCleansUp(t *testing.T) {
	body := []byte("model")
	a := testAsset(body, "file")
	var calls atomic.Int32
	opts := PrepareOptions{CacheDir: t.TempDir()}
	if _, err := prepare(context.Background(), opts, []asset{a}, assetClient([]byte("broken"), &calls)); err == nil {
		t.Fatal("accepted bad checksum")
	}
	entries, err := os.ReadDir(opts.CacheDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.Name() != ".prepare.lock" {
			t.Fatalf("left partial installation: %s", e.Name())
		}
	}
	if _, err := prepare(context.Background(), opts, []asset{a}, assetClient(body, &calls)); err != nil {
		t.Fatal(err)
	}
}

func TestPrepareCancellationAndConcurrentCalls(t *testing.T) {
	body := []byte("model")
	a := testAsset(body, "file")
	var calls atomic.Int32
	opts := PrepareOptions{CacheDir: t.TempDir()}
	client := assetClient(body, &calls)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := prepare(ctx, opts, []asset{a}, client); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
	var wg sync.WaitGroup
	for range 2 {
		wg.Go(func() {
			if _, err := prepare(context.Background(), opts, []asset{a}, client); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if calls.Load() != 1 {
		t.Fatalf("concurrent prepares downloaded %d times", calls.Load())
	}
}

func TestPrepareLocalArchives(t *testing.T) {
	for _, format := range []string{"tgz", "zip"} {
		t.Run(format, func(t *testing.T) {
			var b bytes.Buffer
			contents := map[string]string{"package/lib/libtest.so.1": "native", "package/LICENSE": "license", "package/include/header.h": "skip"}
			if format == "tgz" {
				gz := gzip.NewWriter(&b)
				tw := tar.NewWriter(gz)
				for name, value := range contents {
					if err := tw.WriteHeader(&tar.Header{Name: name, Size: int64(len(value)), Mode: 0644}); err != nil {
						t.Fatal(err)
					}
					if _, err := tw.Write([]byte(value)); err != nil {
						t.Fatal(err)
					}
				}
				if err := tw.WriteHeader(&tar.Header{Name: "package/lib/libtest.so", Typeflag: tar.TypeSymlink, Linkname: "/outside"}); err != nil {
					t.Fatal(err)
				}
				if err := tw.Close(); err != nil {
					t.Fatal(err)
				}
				if err := gz.Close(); err != nil {
					t.Fatal(err)
				}
			} else {
				zw := zip.NewWriter(&b)
				for name, value := range contents {
					w, err := zw.Create(name)
					if err != nil {
						t.Fatal(err)
					}
					if _, err = w.Write([]byte(value)); err != nil {
						t.Fatal(err)
					}
				}
				if err := zw.Close(); err != nil {
					t.Fatal(err)
				}
			}
			a := testAsset(b.Bytes(), format)
			source := t.TempDir()
			if err := os.WriteFile(filepath.Join(source, "runtime"), b.Bytes(), 0600); err != nil {
				t.Fatal(err)
			}
			var calls atomic.Int32
			dir, err := prepare(context.Background(), PrepareOptions{CacheDir: t.TempDir(), ArchiveDir: source, Offline: true}, []asset{a}, assetClient(nil, &calls))
			if err != nil {
				t.Fatal(err)
			}
			for name, value := range map[string]string{"lib/libtest.so.1": "native", "licenses/runtime/LICENSE": "license"} {
				got, err := os.ReadFile(filepath.Join(dir, name))
				if err != nil || string(got) != value {
					t.Fatalf("%s: %q %v", name, got, err)
				}
			}
			if _, err := os.Lstat(filepath.Join(dir, "lib/libtest.so")); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("created archive symlink")
			}
			if calls.Load() != 0 {
				t.Fatal("local archive caused a download")
			}
		})
	}
}

func TestPrepareRejectsArchiveTraversal(t *testing.T) {
	var b bytes.Buffer
	zw := zip.NewWriter(&b)
	w, err := zw.Create("../lib/escape.so")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = w.Write([]byte("bad")); err != nil {
		t.Fatal(err)
	}
	if err = zw.Close(); err != nil {
		t.Fatal(err)
	}
	a := testAsset(b.Bytes(), "zip")
	var calls atomic.Int32
	_, err = prepare(context.Background(), PrepareOptions{CacheDir: t.TempDir()}, []asset{a}, assetClient(b.Bytes(), &calls))
	if err == nil || !strings.Contains(err.Error(), "unsafe archive path") {
		t.Fatalf("accepted traversal: %v", err)
	}
}

func TestPreparedAssetsRejectUnsupportedModes(t *testing.T) {
	for _, tc := range []struct {
		opts     PrepareOptions
		os, arch string
	}{{PrepareOptions{Fast: true}, "linux", "amd64"}, {PrepareOptions{CUDA: true}, "darwin", "arm64"}, {PrepareOptions{}, "windows", "amd64"}} {
		if _, _, err := preparedAssets(tc.opts, tc.os, tc.arch); err == nil {
			t.Fatalf("accepted unsupported mode: %+v", tc)
		}
	}
}
