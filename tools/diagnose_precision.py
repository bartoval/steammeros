"""Isolate precision loss in SCNet's first DualPathRNN using identical inputs.

    python -I tools/diagnose_precision.py --out data/results/precision-dp0

Writes small stage-specific ONNX graphs and report.json, without replacing the
production model or references. CPU FP32; compares original and rewritten
GroupNorm, with ORT graph optimizations disabled and enabled.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_onnx as export
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch


class OutputOnly(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        result = self.module(x)
        return result[0] if isinstance(result, tuple) else result


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual, expected):
    if actual.shape != expected.shape:
        return {"shape_matches": False, "actual_shape": list(actual.shape)}
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"shape_matches": True, "finite": False}
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    error = float(np.sum(difference**2))
    signal = float(np.sum(expected.astype(np.float64) ** 2))
    return {
        "shape_matches": True,
        "finite": True,
        "exact": error == 0,
        "snr_db": float(10 * np.log10(signal / error)) if error > 0 and signal > 0 else None,
        "max_abs_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=export.TEST_AUDIO)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--stages", nargs="+", choices=["norm0", "lstm0", "linear0", "norm1", "lstm1", "linear1", "block"]
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.threads <= 0:
        parser.error("seconds and threads must be positive")
    export.check_runtime_version()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    model = export.load_scnet()

    audio, rate = sf.read(args.input, dtype="float32", frames=args.seconds * 44100)
    if rate != 44100 or audio.ndim != 2 or audio.shape[1] != 2:
        parser.error("input must be stereo at 44100 Hz")
    audio = torch.from_numpy(audio.T.copy())
    mono = audio.mean(0)
    if len(mono) < 4096 or not torch.isfinite(audio).all() or mono.std() == 0:
        parser.error("input must contain finite, nonconstant audio of at least 4096 frames")
    audio = (audio - mono.mean()) / mono.std()

    stages = {
        "norm0": model.separation_net.dp_modules[0].norm_layers[0],
        "lstm0": model.separation_net.dp_modules[0].lstm_layers[0],
        "linear0": model.separation_net.dp_modules[0].linear_layers[0],
        "norm1": model.separation_net.dp_modules[0].norm_layers[1],
        "lstm1": model.separation_net.dp_modules[0].lstm_layers[1],
        "linear1": model.separation_net.dp_modules[0].linear_layers[1],
        "block": model.separation_net.dp_modules[0],
    }
    if args.stages:
        stages = {name: stages[name] for name in args.stages}
    captured = {}

    def capture(name):
        def hook(module, inputs, result):
            output = result[0] if isinstance(result, tuple) else result
            captured[name] = (inputs[0].detach().clone(), output.detach().numpy().copy())

        return hook

    hooks = [module.register_forward_hook(capture(name)) for name, module in stages.items()]
    try:
        with torch.no_grad():
            x = export.spectrum_of(model, audio)
            x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (1e-5 + x.std(dim=(1, 2, 3), keepdim=True))
            for encoder in model.encoder:
                x, _, _, _ = encoder(x)
            model.separation_net.dp_modules[0](x)
    finally:
        for hook in hooks:
            hook.remove()

    report = {
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "scnet_commit": subprocess.check_output(
            ["git", "-C", str(export.ROOT / "third_party" / "SCNet"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint_sha256": checksum(export.MODEL_DIR / "SCNet-large.th"),
        "config_sha256": checksum(export.MODEL_DIR / "config.yaml"),
        "input": str(args.input.resolve()),
        "input_sha256": checksum(args.input),
        "frames": audio.shape[-1],
        "threads": args.threads,
        "opset": export.OPSET,
        "items": [],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, module in stages.items():
        stage_input, expected = captured[name]
        variants = ["original", "two-pass"] if name.startswith("norm") or name == "block" else ["original"]
        for variant in variants:
            working = copy.deepcopy(module).eval()
            wrapper = OutputOnly(working).eval()
            context = export.export_rewrites(working) if variant == "two-pass" else nullcontext()
            path = args.out / f"{name}-{variant}.onnx"
            with context, torch.no_grad():
                rewritten = metrics(wrapper(stage_input).numpy(), expected)
                torch.onnx.export(
                    wrapper,
                    (stage_input,),
                    path,
                    opset_version=export.OPSET,
                    dynamo=True,
                    verbose=False,
                    external_data=False,
                    input_names=["input"],
                    output_names=["output"],
                )
            for label, level in [
                ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
            ]:
                options = ort.SessionOptions()
                options.intra_op_num_threads = args.threads
                options.inter_op_num_threads = 1
                options.graph_optimization_level = level
                session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                actual = session.run(["output"], {"input": stage_input.numpy()})[0]
                result = metrics(actual, expected)
                report["items"].append(
                    {
                        "stage": name,
                        "variant": variant,
                        "optimizations": label,
                        "shape": list(expected.shape),
                        "pytorch_rewrite": rewritten,
                        **result,
                    }
                )
                (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
                print(f"{name:8s} {variant:9s} {label:8s} {result}", flush=True)
                del session
    return 0


if __name__ == "__main__":
    sys.exit(main())
