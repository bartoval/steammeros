package stems

import (
	"strings"
	"testing"
)

func TestVersion(t *testing.T) {
	if got := NativeAPIVersion(); got != APIVersion {
		t.Fatalf("native API version %d, binding expects %d", got, APIVersion)
	}
	if v := Version(); !strings.HasPrefix(v, "stems ") {
		t.Fatalf("unexpected version string %q", v)
	}
}
