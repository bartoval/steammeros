package stems

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

// PrepareOptions selects the assets to install. The zero value prepares CPU inference.
type PrepareOptions struct {
	CUDA       bool   // NVIDIA GPU; Linux amd64. Installs CUDA/cuDNN runtimes, not the driver.
	Fast       bool   // Use the optional mixed-precision model. Requires CUDA.
	CacheDir   string // Empty: os.UserCacheDir()/steammeros. May be an application bundle directory.
	Offline    bool   // Never contact a server. Reuse the cache or install from ArchiveDir.
	ArchiveDir string // Optional directory containing the original release archives/model files.
}

// Prepare installs pinned model/runtime assets and returns options for NewEngine.
// Only this explicit call may download files; NewEngine and Separate never do.
// Existing installations are verified and reused. The returned paths are absolute.
// Prepare does not load native libraries, change environment variables or need Python.
func Prepare(ctx context.Context, opts PrepareOptions) (Options, error) {
	assets, library, err := preparedAssets(opts, runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return Options{}, err
	}
	dir, err := prepare(ctx, opts, assets, http.DefaultClient)
	if err != nil {
		return Options{}, err
	}
	model := "scnet-large.onnx"
	result := Options{CUDA: opts.CUDA, ORTLibrary: filepath.Join(dir, "lib", library)}
	if opts.Fast {
		model = "scnet-large-cuda.onnx"
		result.CUDAOpsLibrary = filepath.Join(dir, "lib", "libstems_cuda_ops.so")
	}
	result.ModelPath = filepath.Join(dir, model)
	return result, nil
}

type asset struct {
	URL    string
	SHA256 string
	Format string // file, tgz, zip
	Target string // destination for an unarchived file
}

type installedFiles map[string]string // relative filename -> SHA-256

func prepare(ctx context.Context, opts PrepareOptions, assets []asset, client *http.Client) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	cache := opts.CacheDir
	if cache == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return "", err
		}
		cache = filepath.Join(base, "steammeros")
	}
	cache, err := filepath.Abs(cache)
	if err != nil {
		return "", err
	}
	plan, _ := json.Marshal(assets)
	key := sha256.Sum256(plan)
	dest := filepath.Join(cache, fmt.Sprintf("assets-v1-%x", key[:12]))
	// A complete cache also works from a read-only application installation.
	if ok, err := verifyInstallation(ctx, dest); ok || err != nil {
		return dest, err
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return "", err
	}
	lock, err := os.OpenFile(filepath.Join(cache, ".prepare.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return "", err
	}
	defer lock.Close()
	for {
		err = syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			break
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) {
			return "", err
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
	defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)
	if ok, err := verifyInstallation(ctx, dest); ok || err != nil {
		return dest, err
	}
	stage, err := os.MkdirTemp(cache, ".prepare-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(stage)
	files := installedFiles{}
	for _, item := range assets {
		if err := installAsset(ctx, stage, opts, item, client, files); err != nil {
			return "", fmt.Errorf("prepare %s: %w", path.Base(item.URL), err)
		}
	}
	if err := saveFile(ctx, stage, "licenses/SCNet.txt", strings.NewReader(scnetLicense), files); err != nil {
		return "", err
	}
	encoded, err := json.Marshal(files)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(stage, ".complete.json"), encoded, 0o644); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.Rename(stage, dest); err != nil {
		return "", err
	}
	return dest, nil
}

func verifyInstallation(ctx context.Context, dir string) (bool, error) {
	data, err := os.ReadFile(filepath.Join(dir, ".complete.json"))
	if errors.Is(err, os.ErrNotExist) {
		if _, e := os.Stat(dir); errors.Is(e, os.ErrNotExist) {
			return false, nil
		}
	}
	corrupt := func(err error) (bool, error) {
		return false, fmt.Errorf("incomplete or damaged assets in %s; remove that directory and prepare again: %w", dir, err)
	}
	if err != nil {
		return corrupt(err)
	}
	var files installedFiles
	if err := json.Unmarshal(data, &files); err != nil {
		return corrupt(err)
	}
	if len(files) == 0 {
		return corrupt(errors.New("empty file manifest"))
	}
	for name, expected := range files {
		if err := safeName(name); err != nil {
			return corrupt(err)
		}
		p := filepath.Join(dir, filepath.FromSlash(name))
		info, err := os.Lstat(p)
		if err != nil {
			return corrupt(err)
		}
		if !info.Mode().IsRegular() {
			return corrupt(fmt.Errorf("not a regular file: %s", name))
		}
		f, err := os.Open(p)
		if err != nil {
			return corrupt(err)
		}
		got, err := digest(ctx, f)
		f.Close()
		if err != nil {
			return false, err
		}
		if got != expected {
			return corrupt(fmt.Errorf("checksum mismatch: %s", name))
		}
	}
	return true, nil
}

// contextReader also makes hashing and extraction cancellable, not only HTTP requests.
type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}
func digest(ctx context.Context, reader io.Reader) (string, error) {
	h := sha256.New()
	if _, err := io.Copy(h, contextReader{ctx, reader}); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func installAsset(ctx context.Context, dir string, opts PrepareOptions, item asset, client *http.Client, files installedFiles) error {
	var source io.ReadCloser
	var err error
	if opts.ArchiveDir != "" {
		source, err = os.Open(filepath.Join(opts.ArchiveDir, path.Base(item.URL)))
		if err != nil {
			return err
		} // An explicit local source never falls back to the network.
	} else {
		if opts.Offline {
			return errors.New("assets are not cached; run Prepare with Offline false once, or provide ArchiveDir")
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, item.URL, nil)
		if err != nil {
			return err
		}
		response, err := client.Do(req)
		if err != nil {
			return err
		}
		if response.StatusCode != http.StatusOK {
			response.Body.Close()
			return fmt.Errorf("HTTP %s", response.Status)
		}
		source = response.Body
	}
	defer source.Close()
	archive, err := os.CreateTemp(dir, ".asset-")
	if err != nil {
		return err
	}
	defer os.Remove(archive.Name())
	h := sha256.New()
	_, copyErr := io.Copy(io.MultiWriter(archive, h), contextReader{ctx, source})
	closeErr := archive.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	if hex.EncodeToString(h.Sum(nil)) != item.SHA256 {
		return errors.New("download checksum mismatch")
	}
	if item.Format == "file" {
		f, err := os.Open(archive.Name())
		if err != nil {
			return err
		}
		defer f.Close()
		return saveFile(ctx, dir, item.Target, f, files)
	}
	return unpack(ctx, archive.Name(), dir, item, files)
}

// Archives are never extracted as a tree. Only regular runtime libraries and
// license files are copied to controlled destinations; links are not followed.
func memberTarget(name, label string) (string, error) {
	if err := safeName(name); err != nil {
		return "", err
	}
	base := path.Base(name)
	if strings.Contains(name, "/lib/") && (strings.Contains(base, ".so") || strings.HasSuffix(base, ".dylib")) {
		return "lib/" + base, nil
	}
	lower := strings.ToLower(base)
	if strings.Contains(lower, "license") || strings.Contains(lower, "notice") {
		return "licenses/" + label + "/" + base, nil
	}
	return "", nil
}
func safeName(name string) error {
	if name == "" || strings.Contains(name, "\\") || strings.HasPrefix(name, "/") {
		return fmt.Errorf("unsafe archive path: %q", name)
	}
	for _, part := range strings.Split(name, "/") {
		if part == ".." {
			return fmt.Errorf("unsafe archive path: %q", name)
		}
	}
	return nil
}
func saveFile(ctx context.Context, dir, name string, reader io.Reader, files installedFiles) error {
	if err := safeName(name); err != nil {
		return err
	}
	if _, exists := files[name]; exists {
		return fmt.Errorf("duplicate asset file: %s", name)
	}
	target := filepath.Join(dir, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	h := sha256.New()
	_, copyErr := io.Copy(io.MultiWriter(f, h), contextReader{ctx, reader})
	closeErr := f.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	files[name] = hex.EncodeToString(h.Sum(nil))
	return nil
}
func unpack(ctx context.Context, archive, dir string, item asset, files installedFiles) error {
	label := strings.Split(path.Base(item.URL), ".")[0]
	count := 0
	write := func(name string, reader io.Reader) error {
		target, err := memberTarget(name, label)
		if err != nil {
			return err
		}
		if target == "" {
			return nil
		}
		count++
		return saveFile(ctx, dir, target, reader, files)
	}
	switch item.Format {
	case "tgz":
		f, err := os.Open(archive)
		if err != nil {
			return err
		}
		defer f.Close()
		gz, err := gzip.NewReader(f)
		if err != nil {
			return err
		}
		defer gz.Close()
		tr := tar.NewReader(contextReader{ctx, gz})
		for {
			header, err := tr.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
			if err := safeName(header.Name); err != nil {
				return err
			}
			if header.Typeflag == tar.TypeReg {
				if err := write(header.Name, tr); err != nil {
					return err
				}
			}
		}
	case "zip":
		z, err := zip.OpenReader(archive)
		if err != nil {
			return err
		}
		defer z.Close()
		for _, entry := range z.File {
			if err := safeName(entry.Name); err != nil {
				return err
			}
			if !entry.Mode().IsRegular() {
				continue
			}
			target, err := memberTarget(entry.Name, label)
			if err != nil {
				return err
			}
			if target == "" {
				continue
			}
			f, err := entry.Open()
			if err != nil {
				return err
			}
			err = write(entry.Name, f)
			f.Close()
			if err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unsupported archive format %q", item.Format)
	}
	if count == 0 {
		return errors.New("archive contains no runtime assets")
	}
	return nil
}
