"""Pick 25 s test excerpts from the MUSDB18-HQ tracks and cut them out (mixture + true stems).

python -I tools/make_excerpts.py            # reads data/musdb18hq/test, writes data/excerpts
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = ROOT / "data" / "musdb18hq" / "test"
OUT_DIR = ROOT / "data" / "excerpts"

STEMS = ["drums", "bass", "other", "vocals"]
SAMPLERATE = 44100
BLOCK = 4410  # 0.1 s: features are measured per block, then averaged per window
WINDOW_BLOCKS = 250  # 25 s excerpts
HOP_BLOCKS = 25  # candidate windows start every 2.5 s
HF_CUTOFF_HZ = 6000  # "cymbal" energy = drums energy above this
EPS = 1e-12

# Three full tracks for timing/memory tests: short, typical and long.
FULL_TRACKS = ["PR - Oh No", "Timboz - Pony", "Georgia Wonder - Siren"]


def db(power):
    return 10 * np.log10(power + EPS)


# Per-block powers for one track. Everything we score later is built from these.
# Returns a dict of 1-D arrays, one value per 0.1 s block.
def block_features(track_dir: Path) -> dict:
    audio = {name: sf.read(track_dir / f"{name}.wav", dtype="float64")[0] for name in ["mixture"] + STEMS}
    blocks = len(audio["mixture"]) // BLOCK

    def per_block(x):
        # (samples, channels) -> (blocks, BLOCK, channels), drop the tail that doesn't fill a block
        return x[: blocks * BLOCK].reshape(blocks, BLOCK, -1)

    feats = {}
    for name, x in audio.items():
        feats[name] = (per_block(x) ** 2).mean(axis=(1, 2))

    # Stereo width of the vocals: side energy vs mid energy. Wide = doubled/backing vocals, choirs.
    vocals = per_block(audio["vocals"])
    feats["vocals_mid"] = (((vocals[..., 0] + vocals[..., 1]) / 2) ** 2).mean(axis=1)
    feats["vocals_side"] = (((vocals[..., 0] - vocals[..., 1]) / 2) ** 2).mean(axis=1)

    # High-frequency share of the drums (cymbals, hi-hats), from the spectrum of each block.
    drums_mono = per_block(audio["drums"]).mean(axis=2)
    spectrum = np.abs(np.fft.rfft(drums_mono, axis=1)) ** 2
    hf_bin = int(HF_CUTOFF_HZ * BLOCK / SAMPLERATE)
    feats["drums_hf"] = spectrum[:, hf_bin:].sum(axis=1)
    feats["drums_all"] = spectrum.sum(axis=1)

    # Block level of the bass in dB, used to measure how steady (sustained) it is.
    feats["bass_db"] = db(feats["bass"])
    return feats


# Averages each block feature over every candidate 25 s window.
# Also returns the std of the bass level inside each window.
def window_features(feats: dict) -> dict:
    blocks = len(feats["mixture"])
    starts = np.arange(0, blocks - WINDOW_BLOCKS + 1, HOP_BLOCKS)

    def window_mean(x):
        csum = np.concatenate([[0], np.cumsum(x)])
        return (csum[starts + WINDOW_BLOCKS] - csum[starts]) / WINDOW_BLOCKS

    win = {name: window_mean(values) for name, values in feats.items()}
    win["bass_db_std"] = np.sqrt(np.maximum(window_mean(feats["bass_db"] ** 2) - win["bass_db"] ** 2, 0))
    win["start_block"] = starts
    return win


# Scores for each category. Higher = better example. -inf = window not eligible.
# All levels are relative to the mixture, in dB, so loud and quiet tracks compare fairly.
def category_scores(win: dict) -> dict:
    mix = win["mixture"]
    rel = {name: db(win[name]) - db(mix) for name in STEMS}
    accompaniment = db(win["drums"] + win["bass"] + win["other"]) - db(mix)
    vocals_width = db(win["vocals_side"]) - db(win["vocals_mid"])
    drums_hf = db(win["drums_hf"]) - db(win["drums_all"])
    no = -np.inf

    return {
        # all four stems clearly present
        "dense_mix": np.minimum.reduce([rel[s] for s in STEMS]),
        # vocals well above everything else
        "solo_voice": rel["vocals"] - accompaniment,
        # vocals present and spread wide in stereo
        "wide_vocals": np.where(rel["vocals"] > -15, vocals_width, no),
        # strong bass whose level barely moves
        "sustained_bass": np.where(rel["bass"] > -12, -win["bass_db_std"], no),
        # drums present with lots of energy above 6 kHz
        "cymbals": np.where(rel["drums"] > -15, drums_hf, no),
        # quietest passage that is not digital silence
        "near_silent": np.where(db(mix) > -80, -db(mix), no),
        # music playing but (almost) no vocals: does the vocals stem stay empty?
        "instrumental": np.where(
            rel["vocals"] < -40, np.minimum.reduce([rel[s] for s in ["drums", "bass", "other"]]), no
        ),
        # drums dominate the mix
        "drums_forward": rel["drums"],
        # "other" (guitars, keys, strings...) dominates the mix
        "other_forward": rel["other"],
    }


# Three excerpts are chosen by track rather than by a measurement, because "electronic",
# "acoustic" or "reverb" can't be measured reliably from levels. Genre is a guess from the
# title/artist, NOT verified by listening. Inside the track we take the densest window.
BY_TRACK = {
    "electronic_guess": "AM Contra - Heart Peripheral",
    "acoustic_guess": "Cristina Vane - So Easy",
    "cinematic_reverb_guess": "Skelpolu - Resurrection",
}


# Picks one window per category, greedily, never using the same track twice
# so the excerpts cover as many different songs as possible.
# Rare categories go first (e.g. only a few tracks have an instrumental passage),
# otherwise a common category could grab the only track that fits a rare one.
def choose_excerpts(scores_by_track: dict, windows_by_track: dict) -> list:
    used = set(BY_TRACK.values())
    picks = []

    for category, track in BY_TRACK.items():
        best = int(np.argmax(scores_by_track[track]["dense_mix"]))
        picks.append((category, track, best, float(scores_by_track[track]["dense_mix"][best])))

    def tracks_eligible(category):
        return sum(
            np.isfinite(scores[category]).any()
            for track, scores in scores_by_track.items()
            if track not in used
        )

    categories = list(scores_by_track[next(iter(scores_by_track))])
    for category in sorted(categories, key=tracks_eligible):
        best_track, best_index, best_score = None, None, -np.inf
        for track, scores in scores_by_track.items():
            if track in used:
                continue
            index = int(np.argmax(scores[category]))
            if scores[category][index] > best_score:
                best_track, best_index, best_score = track, index, scores[category][index]
        if best_track is None:
            sys.exit(f"no eligible window for {category}")
        used.add(best_track)
        picks.append((category, best_track, best_index, float(best_score)))

    excerpts = []
    for number, (category, track, index, score) in enumerate(picks, start=1):
        start = int(windows_by_track[track]["start_block"][index]) * BLOCK
        excerpts.append(
            {
                "id": f"{number:02d}-{category}",
                "category": category,
                "track": track,
                "start_s": start / SAMPLERATE,
                "length_s": WINDOW_BLOCKS * BLOCK / SAMPLERATE,
                "score": round(score, 2),
                "basis": "track choice, genre unverified"
                if category in BY_TRACK
                else "measured on true stems",
            }
        )
    return excerpts


# Copies the mixture and the four true stems for one excerpt. Keeps 16-bit PCM:
# the source files are 16-bit, so this is an exact copy of those samples.
def cut_excerpt(excerpt: dict) -> None:
    out = OUT_DIR / excerpt["id"]
    out.mkdir(parents=True, exist_ok=True)
    start = round(excerpt["start_s"] * SAMPLERATE)
    frames = round(excerpt["length_s"] * SAMPLERATE)

    for name in ["mixture"] + STEMS:
        audio, _ = sf.read(
            TRACKS_DIR / excerpt["track"] / f"{name}.wav", start=start, frames=frames, dtype="int16"
        )
        sf.write(out / f"{name}.wav", audio, SAMPLERATE, subtype="PCM_16")


def main() -> int:
    scores_by_track = {}
    windows_by_track = {}
    for track_dir in sorted(TRACKS_DIR.iterdir()):
        windows = window_features(block_features(track_dir))
        windows_by_track[track_dir.name] = windows
        scores_by_track[track_dir.name] = category_scores(windows)
        print(f"measured {track_dir.name}", flush=True)

    excerpts = choose_excerpts(scores_by_track, windows_by_track)
    for excerpt in excerpts:
        cut_excerpt(excerpt)
        print(
            f"{excerpt['id']:28s} {excerpt['track']:45s} @ {excerpt['start_s']:6.1f}s  score {excerpt['score']}"
        )

    manifest = {"excerpts": excerpts, "full_tracks": FULL_TRACKS}
    (OUT_DIR / "excerpts.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
