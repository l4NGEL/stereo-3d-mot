#!/usr/bin/env python3
"""Generate a tiny ONNX model with the YOLOv8 detection-head signature.

It ignores the pixel values and emits a fixed output tensor holding three raw
detections, so the C++ OnnxDetector's session handling, output transpose,
confidence filter, letterbox inversion and NMS can be unit tested without a
real (multi-MB, non-deterministic) network or any download.

The anchor count is shrunk to 300 (vs 8400 for a real 640px YOLOv8) purely to
keep the committed fixture small; the parse code reads the shape dynamically.

    input   images   float32 [1, 3, 640, 640]
    output  output0  float32 [1, 84, 300]    (4 bbox + 80 class scores, YOLOv8)

Baked detections (raw, letterboxed 640x640 pixel space):
    col 20:   cx=200 cy=160 w=80  h=120   class 0  score 0.92
    col 40:   cx=450 cy=300 w=100 h=90    class 2  score 0.75
    col 41:   cx=452 cy=302 w=98  h=92    class 2  score 0.70   <- NMS duplicate

Requires:  pip install onnx numpy
Run:       python scripts/make_test_model.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper

NUM_ANCHORS = 300
NUM_CLASSES = 80
ROWS = 4 + NUM_CLASSES

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "tiny_yolov8.onnx"


def build_output_constant() -> np.ndarray:
    t = np.zeros((1, ROWS, NUM_ANCHORS), dtype=np.float32)

    def put(col, cx, cy, w, h, cls, score):
        t[0, 0, col] = cx
        t[0, 1, col] = cy
        t[0, 2, col] = w
        t[0, 3, col] = h
        t[0, 4 + cls, col] = score

    put(20, 200.0, 160.0, 80.0, 120.0, cls=0, score=0.92)
    put(40, 450.0, 300.0, 100.0, 90.0, cls=2, score=0.75)
    put(41, 452.0, 302.0, 98.0, 92.0, cls=2, score=0.70)
    return t


def main() -> None:
    const = build_output_constant()

    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output0 = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, ROWS, NUM_ANCHORS])

    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["det_const"],
        value=helper.make_tensor(
            "det_const", TensorProto.FLOAT, const.shape, const.flatten().tolist()
        ),
    )
    # Keep `images` live in the graph (mean * 0), so the model has a real input.
    mean_node = helper.make_node("ReduceMean", inputs=["images"], outputs=["img_mean"], keepdims=0)
    zero_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero"],
        value=helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
    )
    scaled_node = helper.make_node("Mul", inputs=["img_mean", "zero"], outputs=["img_zero"])
    add_node = helper.make_node("Add", inputs=["det_const", "img_zero"], outputs=["output0"])

    graph = helper.make_graph(
        [const_node, mean_node, zero_node, scaled_node, add_node],
        "tiny_yolov8",
        [images],
        [output0],
    )
    model = helper.make_model(
        graph, producer_name="stereo-3d-mot", opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT))
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
