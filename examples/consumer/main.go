// A complete consumer: no checkout-relative paths or external processes.
// Run once with -prepare to allow downloads, then run without it offline.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"

	"github.com/bartoval/steammeros/stems"
)

func main() {
	download := flag.Bool("prepare", false, "allow downloading pinned assets")
	cuda := flag.Bool("cuda", false, "use NVIDIA GPU")
	fast := flag.Bool("fast", false, "use mixed precision (requires -cuda)")
	cache := flag.String("cache", "", "asset cache directory")
	archives := flag.String("archives", "", "local release archives, for offline installation")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	opts, err := stems.Prepare(ctx, stems.PrepareOptions{
		CUDA: *cuda, Fast: *fast, CacheDir: *cache, ArchiveDir: *archives, Offline: !*download,
	})
	if err != nil {
		log.Fatal(err)
	}
	engine, err := stems.NewEngine(opts)
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()
	// Replace this one-second tone with decoded mono/stereo audio from your app.
	audio := [][]float32{make([]float32, 44100)}
	for i := range audio[0] {
		audio[0][i] = float32(0.1 * math.Sin(2*math.Pi*440*float64(i)/44100))
	}
	result, err := engine.Separate(ctx, audio, 44100, nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, stem := range result {
		fmt.Printf("%s: %d channels, %d samples\n", stem.Name, len(stem.Channels), len(stem.Channels[0]))
	}
}
