import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import export_onnx as export
import onnxruntime as ort
import torch


class GroupNormPrecisionTest(unittest.TestCase):
    def test_realistic_group_size_preserves_precision(self):
        export.check_runtime_version()
        generator = torch.Generator().manual_seed(1234)
        # The first DualPathRNN reduces groups of this size, rather than small
        # random tensors that hide drift in a long float32 reduction.
        x = torch.randn(1, 256, 85, 432, generator=generator)
        norm = torch.nn.GroupNorm(1, 256).eval()
        with torch.no_grad():
            expected = norm(x).numpy()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "group-norm.onnx"
            with export.export_rewrites(norm), torch.no_grad():
                torch.onnx.export(
                    norm,
                    (x,),
                    path,
                    opset_version=export.OPSET,
                    dynamo=True,
                    verbose=False,
                    external_data=False,
                    input_names=["input"],
                    output_names=["output"],
                )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            actual = session.run(["output"], {"input": x.numpy()})[0]
        self.assertGreaterEqual(export.snr(actual, expected), 110.0)
        # Export rewrites must leave the reference module unchanged afterwards.
        with torch.no_grad():
            self.assertTrue(torch.equal(norm(x), torch.from_numpy(expected)))


if __name__ == "__main__":
    unittest.main()
