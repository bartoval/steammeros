"""Export the SCNet network (spectrum in, spectrum out) to ONNX and check it against PyTorch.

    python -I tools/export_onnx.py      # writes models/scnet-large/scnet-large.onnx
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "SCNet"))

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from ml_collections import ConfigDict

from scnet.SCNet import SCNet
from scnet.separation import FeatureConversion
from scnet.utils import load_model

ort.disable_telemetry_events()

MODEL_DIR = ROOT / "models" / "scnet-large"
OUTPUT = MODEL_DIR / "scnet-large.onnx"
TEST_AUDIO = ROOT / "data" / "excerpts" / "05-dense_mix" / "mixture.wav"
OPSET = 20


# The part of SCNet.forward() between torch.stft and torch.istft, unchanged.
# STFT/iSTFT stay outside the graph (done in C++): ONNX has no iSTFT and the
# padding logic around them is plain index math.
#
# Input:  (batch, 4, 2049, frames)  channels = [left re, left im, right re, right im]
# Output: (batch, 16, 2049, frames) channels = 4 * source + 2 * channel + (0 re, 1 im),
#         sources in model order: drums, bass, other, vocals
class SpectrumNet(torch.nn.Module):

    def __init__(self, model: SCNet, stable_normalization: bool = False):
        super().__init__()
        self.model = model
        self.stable_normalization = stable_normalization

    def forward(self, x):
        m = self.model
        if self.stable_normalization:
            mean, std = precise_mean_std(x)
        else:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        save_skip, save_lengths, save_original_lengths = deque(), deque(), deque()
        for sd_layer in m.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        x = m.separation_net(x)

        for fusion_layer, su_layer in m.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        return x * std + mean


# Same as x.mean(dim=(1,2,3)) and x.std(dim=(1,2,3)) (unbiased), but reducing one axis at a
# time to avoid a single reduction over millions of float32 values. Its effect
# on the full model must be measured against the original reduction.
def precise_mean_std(x):
    count = x.shape[1] * x.shape[2] * x.shape[3]
    mean = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
    centered = x - mean
    squares = (centered * centered).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
    std = torch.sqrt(squares * count / (count - 1))
    return mean, std


# Cosine and sine tables of the length-n DFT, shape (n, n // 2 + 1), scaled for norm="ortho".
# n * k can be huge, so we reduce it modulo n with integers first: the float angle then
# stays in [0, 2pi) and cos/sin are accurate to float32 precision.
def dft_tables(n):
    time_index = torch.arange(n)
    bins = torch.arange(n // 2 + 1)
    product = time_index[:, None] * bins[None, :]
    # (the exporter can't do `% n` with a dynamic n, so the modulo is spelled out with tensors)
    n_tensor = torch.full_like(product, n)
    phase = product - torch.div(product, n_tensor, rounding_mode="floor") * n_tensor
    angle = phase.to(torch.float32) * (2 * math.pi / n)
    scale = 1.0 / torch.sqrt(torch.tensor(n, dtype=torch.float32))
    return torch.cos(angle) * scale, torch.sin(angle) * scale


# Drop-in replacement for FeatureConversion.forward (rfft / irfft over the frame axis).
# Same math, written as matrix products: ONNX Runtime's DFT operator is slow (half the
# runtime) and loses precision (~83 dB vs ~138 dB for a real FFT) on non-power-of-2
# lengths, while its MatMul is fast and accurate.
def feature_conversion_as_matmul(self, x):
    if not self.inverse:
        # rfft: X[k] = sum_n x[n] * (cos - i sin)
        cos, sin = dft_tables(x.shape[3])
        return torch.cat([x @ cos, -(x @ sin)], dim=1)

    # irfft (default length 2 * (bins - 1)): bins 1..n/2-1 count twice (their mirror images),
    # DC and Nyquist once. Their imaginary parts drop out because sin is 0 there.
    half = self.channels // 2
    real, imag = x[:, :half], x[:, half:]
    n = 2 * (x.shape[3] - 1)
    cos, sin = dft_tables(n)
    weight = torch.full((n // 2 + 1,), 2.0)
    weight[0] = 1.0
    weight[-1] = 1.0
    return real @ (cos * weight).T - imag @ (sin * weight).T


# Two-pass GroupNorm with separate reductions for each axis. Flattening an entire
# group into one long axis still loses precision in ORT on real model tensors.
def group_norm_two_pass(self, x):
    shape = x.shape
    groups = x.reshape(shape[0], self.num_groups, shape[1] // self.num_groups, *shape[2:])
    mean = groups
    for axis in range(groups.dim() - 1, 1, -1):
        mean = mean.mean(dim=axis, keepdim=True)
    centered = groups - mean
    variance = centered * centered
    for axis in range(groups.dim() - 1, 1, -1):
        variance = variance.mean(dim=axis, keepdim=True)
    normalized = (centered / torch.sqrt(variance + self.eps)).reshape(shape)

    affine_shape = (1, -1) + (1,) * (x.dim() - 2)
    return normalized * self.weight.view(affine_shape) + self.bias.view(affine_shape)


@contextmanager
def export_rewrites(model):
    # Patch only this model, then restore it so later references stay original.
    saved = []
    try:
        for module in model.modules():
            replacement = None
            if isinstance(module, FeatureConversion):
                replacement = feature_conversion_as_matmul
            elif isinstance(module, torch.nn.GroupNorm):
                replacement = group_norm_two_pass
            if replacement is not None:
                saved.append((module, module.__dict__.get("forward")))
                module.forward = MethodType(replacement, module)
        yield
    finally:
        for module, previous in reversed(saved):
            if previous is None:
                del module.forward
            else:
                module.forward = previous


def load_scnet() -> SCNet:
    config = ConfigDict(yaml.safe_load((MODEL_DIR / "config.yaml").read_text()))
    model = SCNet(**config.model)
    load_model(model, MODEL_DIR / "SCNet-large.th")
    model.eval()
    return model


# Same steps as the start of SCNet.forward(): pad so the frame count is odd, STFT,
# complex -> 4 real channels. Used here only to make test inputs.
def spectrum_of(model: SCNet, audio: torch.Tensor) -> torch.Tensor:
    hop = model.hop_length
    padding = hop - audio.shape[-1] % hop
    if (audio.shape[-1] + padding) // hop % 2 == 0:
        padding += hop
    audio = F.pad(audio, (0, padding))
    spec = torch.stft(audio, **model.stft_config, return_complex=True)
    spec = torch.view_as_real(spec).permute(0, 3, 1, 2).reshape(1, 4, spec.shape[1], spec.shape[2])
    return spec


# SNR of the difference, in dB: how far below the reference the error sits.
def snr(estimate, reference) -> float:
    reference = reference.astype(np.float64)
    error = estimate.astype(np.float64) - reference
    return float(10 * np.log10(np.sum(reference ** 2) / np.sum(error ** 2)))


def check_runtime_version() -> str:
    expected_ort_version = next(
        line.removeprefix("onnxruntime==")
        for line in (ROOT / "tools" / "requirements.lock").read_text().splitlines()
        if line.startswith("onnxruntime==")
    )
    if ort.__version__ != expected_ort_version:
        sys.exit(
            f"ONNX Runtime version mismatch: expected {expected_ort_version}, found {ort.__version__}. "
            f"Install onnxruntime=={expected_ort_version} in the reference environment before exporting."
        )
    return expected_ort_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    check_runtime_version()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)

    model = load_scnet()
    net = SpectrumNet(model).eval()

    # Reference: the untouched network on 20 s of real music (one full chunk).
    audio, _ = sf.read(TEST_AUDIO, dtype="float32", frames=20 * 44100)
    audio = torch.from_numpy(audio.T.copy())
    mono = audio.mean(0)
    audio = (audio - mono.mean()) / mono.std()
    test = spectrum_of(model, audio)
    with torch.no_grad():
        expected = net(test).numpy()

    net = SpectrumNet(model, stable_normalization=True).eval()
    with export_rewrites(model):
        with torch.no_grad():
            rewrite_snr = snr(net(test).numpy(), expected)
            print(f"export rewrites vs original (PyTorch): SNR {rewrite_snr:.1f} dB", flush=True)

        # Export with a short example; the frame axis is dynamic.
        example = spectrum_of(model, torch.randn(2, 44100 * 3) * 0.1)
        frames = torch.export.Dim("frames", min=8, max=4096)
        torch.onnx.export(
            net, (example,), OUTPUT, opset_version=OPSET, dynamo=True,
            input_names=["spectrum"], output_names=["stems"],
            dynamic_shapes={"x": {3: frames}},
            external_data=False,
        )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size >> 20} MiB)")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(OUTPUT), options, providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    actual = session.run(None, {"spectrum": test.numpy()})[0]
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        sys.exit("invalid exported output: shape mismatch or non-finite values")
    actual_snr = snr(actual, expected)
    report = {
        "torch": torch.__version__, "onnxruntime": ort.__version__, "threads": args.threads,
        "opset": OPSET, "input": str(TEST_AUDIO), "audio_frames": audio.shape[-1],
        "output_shape": list(actual.shape), "pytorch_rewrite_snr_db": rewrite_snr,
        "ort_snr_db": actual_snr,
        "max_abs_error": float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max()),
    }
    OUTPUT.with_suffix(".export.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"ONNX Runtime vs original PyTorch: SNR {actual_snr:.1f} dB "
          f"({time.perf_counter() - start:.1f} s for 20 s of audio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
