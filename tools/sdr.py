"""Separation quality (SDR per stem) of a results folder against the true sources.

    python -I tools/sdr.py data/results/excerpts-deterministic data/excerpts
    python -I tools/sdr.py data/results/excerpts-original data/excerpts \\
        --baseline data/results/excerpts-deterministic
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ["drums", "bass", "other", "vocals"]
DELTA = 1e-7  # MDX'21 SDR definition
SILENT_RMS = 1e-4  # true source quieter than this (-80 dBFS): SDR is meaningless, skip it


# SDR as in the MDX'21 challenge: energy of the true source over energy of the error, in dB.
# One number for the whole stereo signal. Higher is better; +1 dB is a clearly audible step.
def sdr(true: np.ndarray, estimate: np.ndarray) -> float:
    signal = np.sum(true**2) + DELTA
    error = np.sum((true - estimate) ** 2) + DELTA
    return float(10 * np.log10(signal / error))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


# SDR for every stem of every item found in both folders.
# Returns {item: {stem: sdr or None}}; None = true source is (almost) silent there.
def evaluate(estimates_dir: Path, sources_dir: Path) -> dict:
    results = {}
    for item_dir in sorted(p for p in estimates_dir.iterdir() if p.is_dir()):
        source_dir = sources_dir / item_dir.name
        if not source_dir.is_dir():
            print(f"skipping {item_dir.name}: no true sources in {sources_dir}", file=sys.stderr)
            continue

        results[item_dir.name] = {}
        for stem in STEMS:
            true, _ = sf.read(source_dir / f"{stem}.wav", dtype="float64")
            estimate, _ = sf.read(item_dir / f"{stem}.wav", dtype="float64")
            if true.shape != estimate.shape:
                sys.exit(f"{item_dir.name}/{stem}: shape {estimate.shape} != true {true.shape}")
            if not np.isfinite(estimate).all():
                sys.exit(f"{item_dir.name}/{stem}: NaN/Inf in the estimate")

            results[item_dir.name][stem] = sdr(true, estimate) if rms(true) >= SILENT_RMS else None
    return results


def fmt(value) -> str:
    return "   -  " if value is None else f"{value:6.2f}"


# Prints one row per item plus median and worst per stem.
# With a baseline, values are the difference (this - baseline) instead of absolute SDR.
def print_table(results: dict, baseline: dict | None) -> None:
    def value(item, stem):
        v = results[item][stem]
        if baseline is None or v is None:
            return v
        return v - baseline[item][stem]

    print(f"{'item':40s}" + "".join(f"{s:>8s}" for s in STEMS))
    for item in results:
        print(f"{item[:40]:40s}" + "".join(f"  {fmt(value(item, s))}" for s in STEMS))

    for label, reduce in [("median", np.median), ("worst", np.min)]:
        row = []
        for stem in STEMS:
            values = [value(item, stem) for item in results if value(item, stem) is not None]
            row.append(float(reduce(values)) if values else None)
        print(f"{label:40s}" + "".join(f"  {fmt(v)}" for v in row))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("estimates", type=Path, help="folder with <item>/{drums,bass,other,vocals}.wav")
    parser.add_argument("sources", type=Path, help="folder with the true sources, same layout")
    parser.add_argument("--baseline", type=Path, help="another results folder: print the SDR difference")
    args = parser.parse_args()

    results = evaluate(args.estimates, args.sources)
    baseline = evaluate(args.baseline, args.sources) if args.baseline else None
    print_table(results, baseline)

    # Saved next to the estimates, so later runs (e.g. the native port) can be compared.
    (args.estimates / "sdr.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
