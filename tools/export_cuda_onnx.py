"""Derives the fast GPU model from scnet-large.onnx (no PyTorch needed).

Each bidirectional LSTM and the Transpose after it become one PackedLSTM16 operator
(stems/cuda/lstm.cpp): FP16 weights and inputs, FP32 math in cuDNN. Everything else
stays FP32. The result needs a CUDA device and libstems_cuda_ops.so.

python -I tools/export_cuda_onnx.py models/scnet-large/scnet-large.onnx \
    models/scnet-large/scnet-large-cuda.onnx
"""

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

DOMAIN = "com.steammeros.cuda"


def attributes(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def convert(source):
    model = copy.deepcopy(source)
    graph = model.graph
    initializers = {v.name: v for v in graph.initializer}
    producers = {name: node for node in graph.node for name in node.output}
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    cache = {}

    # The exporter builds LSTM weights from initializers through a few ops: fold them.
    def constant(name):
        if name in cache:
            return cache[name]
        if name in initializers:
            value = numpy_helper.to_array(initializers[name])
        else:
            node = producers.get(name)
            if node is None:
                raise ValueError(f"weight depends on a runtime input: {name}")
            attrs = attributes(node)
            values = [constant(n) for n in node.input if n]
            if node.op_type == "Constant":
                value = numpy_helper.to_array(attrs["value"])
            elif node.op_type == "Concat":
                value = np.concatenate(values, axis=attrs["axis"])
            elif node.op_type == "Unsqueeze":
                value = np.expand_dims(values[0], tuple(int(a) for a in values[1]))
            elif node.op_type == "Slice":
                slices = [slice(None)] * values[0].ndim
                axes = values[3] if len(values) > 3 else range(len(values[1]))
                steps = values[4] if len(values) > 4 else [1] * len(values[1])
                for start, end, axis, step in zip(values[1], values[2], axes, steps):
                    slices[int(axis)] = slice(int(start), int(end), int(step))
                value = values[0][tuple(slices)]
            else:
                raise ValueError(f"unsupported weight expression: {node.op_type}")
        cache[name] = value
        return value

    # The operator starts from zero states: accept only initial states that are zero.
    def zero_state(name):
        if not name:
            return True
        if name in initializers:
            return bool(np.all(constant(name) == 0))
        node = producers.get(name)
        if node is None:
            return False
        if node.op_type == "Constant":
            return bool(np.all(constant(name) == 0))
        if node.op_type in ("Slice", "Expand", "Reshape", "Unsqueeze", "Squeeze", "Cast"):
            return zero_state(node.input[0])
        if node.op_type == "ConstantOfShape":
            attrs = attributes(node)
            return "value" not in attrs or bool(np.all(numpy_helper.to_array(attrs["value"]) == 0))
        return False

    replacements, removed, added_values = {}, set(), []
    graph_outputs = {v.name for v in graph.output}
    for node in graph.node:
        if node.op_type != "LSTM" or node.domain:
            continue
        attrs = attributes(node)
        hidden = attrs.get("hidden_size", 0)
        if (
            hidden <= 0
            or attrs.get("direction") != b"bidirectional"
            or attrs.get("layout", 0) != 0
            or attrs.get("input_forget", 0) != 0
            or set(attrs) - {"hidden_size", "direction", "layout", "input_forget"}
        ):
            raise ValueError(f"unsupported LSTM attributes: {node.name}")
        inputs = list(node.input) + [""] * (8 - len(node.input))
        if len(node.output) != 1 or inputs[4] or inputs[7] or not all(zero_state(n) for n in inputs[5:7]):
            raise ValueError(f"LSTM needs default zero states, no sequence lengths or peepholes: {node.name}")
        users = consumers.get(node.output[0], [])
        if len(users) != 1 or node.output[0] in graph_outputs or users[0].op_type != "Transpose":
            raise ValueError(f"LSTM output must feed only its layout transpose: {node.name}")
        transpose = users[0]
        if attributes(transpose).get("perm") != [0, 2, 1, 3]:
            raise ValueError(f"unexpected LSTM output layout: {node.name}")

        names = []
        for i, shape in zip((1, 2, 3), ((2, 4 * hidden, hidden), (2, 4 * hidden, hidden), (2, 8 * hidden))):
            value = constant(inputs[i])
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f"unexpected SCNet weight shape/type: {inputs[i]}")
            value = np.ascontiguousarray(value, dtype=np.float16)
            if not np.isfinite(value).all():
                raise ValueError(f"weight does not fit in FP16: {inputs[i]}")
            name = f"{node.name}_cuda_weight_{i}"
            if name in initializers or name in producers:
                raise ValueError(f"generated name already exists: {name}")
            graph.initializer.append(numpy_helper.from_array(value, name))
            names.append(name)

        # FP32 -> FP16 -> PackedLSTM16 -> FP32, same output name as the Transpose.
        x, y = f"{node.name}_half_input", f"{node.name}_half_output"
        original = next((v for v in graph.value_info if v.name == transpose.output[0]), None)
        if original is not None:
            info = copy.deepcopy(original)
            info.name = y
            info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
            added_values.append(info)
        replacements[node.output[0]] = [
            helper.make_node("Cast", [inputs[0]], [x], to=onnx.TensorProto.FLOAT16),
            helper.make_node(
                "PackedLSTM16", [x, *names], [y], domain=DOMAIN, name=f"{node.name}_cuda", hidden_size=hidden
            ),
            helper.make_node("Cast", [y], list(transpose.output), to=onnx.TensorProto.FLOAT),
        ]
        removed.add(transpose.output[0])
    if not replacements:
        raise ValueError("no supported LSTM found")

    nodes = []
    for node in graph.node:
        if node.output[0] in replacements:
            nodes.extend(replacements[node.output[0]])
        elif node.output[0] not in removed:
            nodes.append(node)
    # Drop what only fed the old LSTMs (weight expressions, zero states), keeping order.
    needed = set(graph_outputs)
    kept = []
    for node in reversed(nodes):
        if any(n in needed for n in node.output):
            kept.append(node)
            needed.update(n for n in node.input if n)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    values = [v for v in list(graph.value_info) + added_values if v.name in needed]
    del graph.value_info[:]
    graph.value_info.extend(values)
    tensors = [v for v in graph.initializer if v.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(tensors)
    if any(op.domain == DOMAIN for op in model.opset_import):
        raise ValueError("source already imports the CUDA operator domain")
    model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    return model, len(replacements)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("output must differ from the source model")
    model, count = convert(onnx.load(args.source))
    with args.source.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    props = {p.key: p.value for p in model.metadata_props}
    props["stems.source_onnx_sha256"] = digest
    helper.set_model_props(model, props)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output}: {count} LSTMs replaced")


if __name__ == "__main__":
    main()
