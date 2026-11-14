"""Run the original SCNet code on WAV files and save the stems (reference for the port).

    python -I tools/run_reference.py --out data/results/reference \\
        data/musdb18hq/test/*/mixture.wav
"""

import argparse
import functools
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

# Hide the GPU before torch gets imported, otherwise the original code grabs it.
# The reference must be CPU float32 so it doesn't depend on the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT / "third_party" / "SCNet"))

import numpy as np
import scnet.inference as original
import soundfile as sf
import torch
import yaml
from ml_collections import ConfigDict
from scnet.SCNet import SCNet


# Makes sure the weights/config are exactly the ones in models/manifest.json.
# Stops the script on a mismatch (corrupted or wrong file). Files that aren't in
# the manifest only get a warning, so you can still try other checkpoints.
def verify_checksums(*paths: Path) -> None:
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())

    expected = {}
    for model in manifest["models"]:
        for entry in model["files"].values():
            expected[MODELS_DIR / entry["path"]] = entry["sha256"]

    for path in paths:
        path = path.resolve()
        if path not in expected:
            print(f"WARNING: {path} is not in the manifest, not verified", file=sys.stderr)
            continue

        with open(path, "rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected[path]:
            sys.exit(f"checksum mismatch for {path}")


# Builds SCNet from the config, loads the weights and wraps it in the original
# Seperator class (yes, it's misspelled in the original code).
def load_separator(checkpoint: Path, config_path: Path):
    config = ConfigDict(yaml.safe_load(config_path.read_text()))
    model = SCNet(**config.model)
    model.eval()

    separator = original.Seperator(model, str(checkpoint))
    if separator.device.type != "cpu":
        sys.exit("the reference must run on CPU")
    return separator


# Name of the output folder for an input file.
# MUSDB tracks are all called mixture.wav, so those use the parent folder name;
# any other file uses its own name without the extension.
def track_name(path: Path) -> str:
    if path.name == "mixture.wav":
        return path.parent.name
    return path.stem


# Writes each stem as a float32 WAV (no rounding to 16-bit, so we can compare exactly).
# Warns if the model produced NaN/Inf, which would be a bug worth looking at.
def save_stems(stems: dict, samplerate: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in stems.items():
        audio = np.asarray(audio, dtype=np.float32)
        if not np.isfinite(audio).all():
            print(f"WARNING: non-finite values in {out_dir.name}/{name}", file=sys.stderr)
        sf.write(out_dir / f"{name}.wav", audio, samplerate, subtype="FLOAT")


# Checks the model files, loads the model once, then separates every input
# and saves <out>/<track>/{drums,bass,other,vocals}.wav plus <out>/report.json.
#
# --mode deterministic: no random time shift, same input -> same output (use this for parity).
# --mode original:      random shift like the original code, with a fixed seed.
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="input WAV files")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--mode", choices=["deterministic", "original"], default="deterministic")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for --mode original")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "scnet-large" / "SCNet-large.th")
    parser.add_argument("--config", type=Path, default=MODELS_DIR / "scnet-large" / "config.yaml")
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    verify_checksums(args.checkpoint, args.config)
    separator = load_separator(args.checkpoint, args.config)

    if args.mode == "deterministic":
        # The original separate_music_file() calls apply_model(..., overlap=0.5)
        # and leaves shifts at its default (1 = random). Swap in a version with shifts=0.
        original.apply_model = functools.partial(original.apply_model, shifts=0)

    report = {
        "mode": args.mode,
        "seed": args.seed if args.mode == "original" else None,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "items": [],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        name = track_name(path)
        audio, samplerate = separator.load_audio(str(path))

        if args.mode == "original":
            # Reseed per file so each file's result doesn't depend on the ones before it.
            random.seed(args.seed)

        start = time.perf_counter()
        stems, _ = separator.separate_music_file(audio, samplerate)
        elapsed = time.perf_counter() - start

        save_stems(stems, samplerate, args.out / name)

        duration = len(audio) / samplerate
        report["items"].append({"name": name, "seconds": duration, "separate_s": elapsed})
        print(f"{name}: {duration:.1f} s of audio in {elapsed:.1f} s (RTF {elapsed / duration:.3f})")

        # Rewritten after every file, so a crash still leaves a usable report.
        (args.out / "report.json").write_text(json.dumps(report, indent=1))

    return 0


if __name__ == "__main__":
    sys.exit(main())
