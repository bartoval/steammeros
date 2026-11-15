import unittest

import export_cuda_onnx as export
import numpy as np
import onnx
from onnx import helper, numpy_helper


def lstm_model(initial_state=None):
    tensors = [
        numpy_helper.from_array(np.arange(32, dtype=np.float32).reshape(2, 8, 2) / 32, "W"),
        numpy_helper.from_array(np.zeros((2, 8, 2), dtype=np.float32), "R"),
        numpy_helper.from_array(np.zeros((2, 16), dtype=np.float32), "B"),
    ]
    inputs = ["X", "W", "R", "B"]
    if initial_state is not None:
        tensors.append(numpy_helper.from_array(initial_state, "state"))
        inputs += ["", "state", "state"]
    nodes = [
        helper.make_node("LSTM", inputs, ["raw"], name="lstm", hidden_size=2, direction="bidirectional"),
        helper.make_node("Transpose", ["raw"], ["Y"], perm=[0, 2, 1, 3]),
    ]
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["sequence", 1, 2])],
        [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["sequence", 1, 2, 2])],
        tensors,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10)


class CUDAExportTest(unittest.TestCase):
    def test_replaces_lstm_keeping_float32_inputs_and_outputs(self):
        original = lstm_model(np.zeros((2, 1, 2), dtype=np.float32))
        converted, count = export.convert(original)
        onnx.checker.check_model(converted)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in converted.graph.node], ["Cast", "PackedLSTM16", "Cast"])
        self.assertEqual(converted.graph.input[0], original.graph.input[0])
        self.assertEqual(converted.graph.output[0], original.graph.output[0])
        self.assertEqual(original.graph.node[0].op_type, "LSTM")  # source untouched
        self.assertNotIn("state", [v.name for v in converted.graph.initializer])
        for before, after in zip(original.graph.initializer[:3], converted.graph.initializer):
            self.assertEqual(after.data_type, onnx.TensorProto.FLOAT16)
            np.testing.assert_array_equal(
                numpy_helper.to_array(before).astype(np.float16), numpy_helper.to_array(after)
            )

    def test_nonzero_initial_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero states"):
            export.convert(lstm_model(np.ones((2, 1, 2), dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
