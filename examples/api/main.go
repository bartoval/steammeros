// Example HTTP API around the stems package: upload a song, follow the job, download stems.
//
//	go run ./examples/api -model models/scnet-large/scnet-large.onnx
//	curl --data-binary @song.mp3 localhost:8080/jobs        # -> {"id":"1","status":"queued",...}
//	curl localhost:8080/jobs/1                               # status and progress
//	curl -o vocals.wav localhost:8080/jobs/1/stems/vocals    # once done
//	curl -X DELETE localhost:8080/jobs/1                     # cancel, delete its files
//
// Uploads are decoded with ffmpeg, so any format it knows works. Separations wait in a
// queue and run one at a time (-parallel): each needs about 7 GB of RAM on CPU.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"sync"

	"github.com/bartoval/steammeros/internal/audiofile"
	"github.com/bartoval/steammeros/stems"
)

// One uploaded song. The exported fields are what GET /jobs/{id} returns.
type job struct {
	ID       string   `json:"id"`
	Status   string   `json:"status"` // queued, running, done, failed or cancelled
	Progress float64  `json:"progress"`
	Error    string   `json:"error,omitempty"`
	Stems    []string `json:"stems,omitempty"` // download paths, once done

	dir    string // input and stem files
	cancel context.CancelFunc
}

type server struct {
	engine *stems.Engine
	dir    string
	slots  chan struct{} // one token per separation allowed to run at once

	mu     sync.Mutex // guards jobs, nextID and every job's exported fields
	jobs   map[string]*job
	nextID int
}

func main() {
	model := flag.String("model", "models/scnet-large/scnet-large.onnx", "path to the .onnx model")
	ort := flag.String("ort", "", "ONNX Runtime shared library (default: $STEMS_ORT_LIBRARY, then system locations)")
	cuda := flag.Int("cuda", -1, "run the model on this GPU (needs a GPU build of ONNX Runtime); -1 = CPU")
	addr := flag.String("addr", "localhost:8080", "listen address")
	parallel := flag.Int("parallel", 1, "separations running at once")
	dir := flag.String("dir", filepath.Join(os.TempDir(), "steammeros-api"), "where uploads and stems go")
	flag.Parse()

	engine, err := stems.NewEngine(stems.Options{
		ModelPath: *model, ORTLibrary: *ort, CUDA: *cuda >= 0, CUDADevice: *cuda,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	// Jobs outlive the request that created them, so they hang off this context:
	// Ctrl-C cancels whatever is still running.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	s := &server{engine: engine, dir: *dir, slots: make(chan struct{}, *parallel), jobs: map[string]*job{}}
	http.HandleFunc("POST /jobs", func(w http.ResponseWriter, r *http.Request) { s.create(ctx, w, r) })
	http.HandleFunc("GET /jobs/{id}", s.status)
	http.HandleFunc("GET /jobs/{id}/stems/{stem}", s.download)
	http.HandleFunc("DELETE /jobs/{id}", s.remove)

	httpServer := &http.Server{Addr: *addr}
	go func() {
		<-ctx.Done()
		httpServer.Shutdown(context.Background())
	}()
	log.Printf("listening on http://%s", *addr)
	if err := httpServer.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

// POST /jobs: saves the uploaded file, queues the separation, answers right away.
func (s *server) create(ctx context.Context, w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.nextID++
	id := strconv.Itoa(s.nextID)
	s.mu.Unlock()

	dir := filepath.Join(s.dir, id)
	if err := saveUpload(r.Body, dir); err != nil {
		os.RemoveAll(dir) // no job is registered, so nothing else would delete it
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	jobCtx, cancel := context.WithCancel(ctx)
	j := &job{ID: id, Status: "queued", dir: dir, cancel: cancel}
	s.mu.Lock()
	s.jobs[id] = j
	s.mu.Unlock()
	go s.run(jobCtx, j)

	w.WriteHeader(http.StatusAccepted)
	s.writeJSON(w, j)
}

// GET /jobs/{id}: status, progress and, once done, where to download the stems.
func (s *server) status(w http.ResponseWriter, r *http.Request) {
	j := s.find(w, r)
	if j == nil {
		return
	}
	s.writeJSON(w, j)
}

// GET /jobs/{id}/stems/{stem}: one stem as a WAV, at the upload's sample rate.
func (s *server) download(w http.ResponseWriter, r *http.Request) {
	j := s.find(w, r)
	if j == nil {
		return
	}
	stem := r.PathValue("stem")
	s.mu.Lock()
	ready := j.Status == "done" && slices.Contains(j.Stems, "/jobs/"+j.ID+"/stems/"+stem)
	s.mu.Unlock()
	if !ready {
		http.Error(w, "no such stem, or the job isn't done", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	http.ServeFile(w, r, filepath.Join(j.dir, stem+".wav"))
}

// DELETE /jobs/{id}: cancels the job if it's still going and deletes its files.
func (s *server) remove(w http.ResponseWriter, r *http.Request) {
	j := s.find(w, r)
	if j == nil {
		return
	}
	s.mu.Lock()
	delete(s.jobs, j.ID)
	finished := j.Status != "queued" && j.Status != "running"
	s.mu.Unlock()

	j.cancel()
	if finished {
		os.RemoveAll(j.dir)
	} // otherwise run() deletes the files once it has stopped
	w.WriteHeader(http.StatusNoContent)
}

// Waits for a free slot, then decodes, separates and writes the stems.
func (s *server) run(ctx context.Context, j *job) {
	defer s.cleanUpIfRemoved(j)

	select {
	case s.slots <- struct{}{}:
		defer func() { <-s.slots }()
	case <-ctx.Done():
		s.finish(j, ctx.Err())
		return
	}
	s.update(j, func() { j.Status = "running" })

	audio, sampleRate, err := audiofile.Decode(ctx, filepath.Join(j.dir, "upload"))
	if err != nil {
		if ctx.Err() != nil {
			err = ctx.Err() // ffmpeg was killed because the job was deleted
		}
		s.finish(j, err)
		return
	}
	reportProgress := func(done float64) { s.update(j, func() { j.Progress = done }) }
	parts, err := s.engine.Separate(ctx, audio, sampleRate, reportProgress)
	if err != nil {
		s.finish(j, err)
		return
	}

	var downloads []string
	for _, part := range parts {
		if ctx.Err() != nil { // deleted while separating: don't write files nobody will fetch
			s.finish(j, ctx.Err())
			return
		}
		if err := audiofile.WriteWav(filepath.Join(j.dir, part.Name+".wav"), part.Channels, sampleRate); err != nil {
			s.finish(j, err)
			return
		}
		downloads = append(downloads, "/jobs/"+j.ID+"/stems/"+part.Name)
	}
	s.update(j, func() { j.Stems = downloads })
	s.finish(j, nil)
}

// Records how a job ended.
func (s *server) finish(j *job, err error) {
	s.update(j, func() {
		switch {
		case err == nil:
			j.Status, j.Progress = "done", 1
		case errors.Is(err, context.Canceled):
			j.Status = "cancelled"
		default:
			j.Status, j.Error = "failed", err.Error()
		}
	})
}

// A job deleted while it was queued or running leaves its files behind: remove them now.
func (s *server) cleanUpIfRemoved(j *job) {
	s.mu.Lock()
	_, kept := s.jobs[j.ID]
	s.mu.Unlock()
	if !kept {
		os.RemoveAll(j.dir)
	}
}

func (s *server) update(j *job, change func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	change()
}

// Looks up the job named in the URL, answering 404 itself when there's none.
func (s *server) find(w http.ResponseWriter, r *http.Request) *job {
	s.mu.Lock()
	j := s.jobs[r.PathValue("id")]
	s.mu.Unlock()
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
	}
	return j
}

func (s *server) writeJSON(w http.ResponseWriter, j *job) {
	s.mu.Lock()
	body, err := json.Marshal(j)
	s.mu.Unlock()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(append(body, '\n'))
}

func saveUpload(body io.Reader, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	file, err := os.Create(filepath.Join(dir, "upload"))
	if err != nil {
		return err
	}
	defer file.Close()
	if _, err := io.Copy(file, body); err != nil {
		return err
	}
	return file.Close()
}
