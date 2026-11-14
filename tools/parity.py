"""How close a results folder is to the reference (e.g. native port vs original PyTorch).

    python -I tools/parity.py data/results/native data/results/excerpts-deterministic
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ["drums", "bass", "other", "vocals"]
SAMPLERATE = 44100
SILENT_RMS = 1e-4       # reference quieter than this (-80 dBFS) counts as near-silent
MIN_SNR_DB = 50.0       # whole stem
MIN_WINDOW_SNR_DB = 40.0  # worst 1 s window: catches local glitches, e.g. at chunk boundaries
MAX_SILENT_ERROR = 1e-4


# SNR of the difference: how far below the reference signal the error sits, in dB.
def snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    error = np.sum((estimate - reference) ** 2)
    if error == 0:
        return float("inf")
    return float(10 * np.log10(np.sum(reference ** 2) / error))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


# Worst SNR over 1 s windows, skipping windows where the reference is near-silent.
def worst_window_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    worst = float("inf")
    for start in range(0, len(reference), SAMPLERATE):
        ref = reference[start:start + SAMPLERATE]
        if rms(ref) >= SILENT_RMS:
            worst = min(worst, snr(ref, estimate[start:start + SAMPLERATE]))
    return worst


# Compares one stem and returns (passed, printable summary).
def compare_stem(reference: np.ndarray, estimate: np.ndarray) -> tuple[bool, str]:
    if reference.shape != estimate.shape:
        return False, f"shape {estimate.shape} != {reference.shape}"
    if not np.isfinite(estimate).all():
        return False, "NaN/Inf"

    max_error = float(np.max(np.abs(estimate - reference))) if reference.size else 0.0
    if rms(reference) < SILENT_RMS:
        return max_error <= MAX_SILENT_ERROR, f"near-silent, max err {max_error:.1e}"

    whole = snr(reference, estimate)
    window = worst_window_snr(reference, estimate)
    passed = whole >= MIN_SNR_DB and window >= MIN_WINDOW_SNR_DB
    return passed, f"SNR {whole:5.1f} dB, worst 1 s {window:5.1f} dB, max err {max_error:.1e}"


def compare_files(reference_path: Path, estimate_path: Path) -> tuple[bool, str]:
    reference, reference_rate = sf.read(reference_path, dtype="float64")
    if not estimate_path.exists():
        return False, "missing"
    estimate, estimate_rate = sf.read(estimate_path, dtype="float64")
    if estimate_rate != reference_rate:
        return False, f"sample rate {estimate_rate} Hz != reference {reference_rate} Hz"
    if reference_rate != SAMPLERATE:
        return False, f"sample rate {reference_rate} Hz != expected {SAMPLERATE} Hz"
    return compare_stem(reference, estimate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("estimates", type=Path, help="folder with <item>/<stem>.wav to check")
    parser.add_argument("reference", type=Path, help="reference folder, same layout")
    args = parser.parse_args()

    failures = 0
    for item_dir in sorted(p for p in args.reference.iterdir() if p.is_dir()):
        for stem in STEMS:
            reference_path = item_dir / f"{stem}.wav"
            estimate_path = args.estimates / item_dir.name / f"{stem}.wav"
            passed, summary = compare_files(reference_path, estimate_path)

            failures += not passed
            print(f"{'ok ' if passed else 'FAIL'} {item_dir.name[:32]:32s} {stem:7s} {summary}")

    print(f"\n{failures} failing stems")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
