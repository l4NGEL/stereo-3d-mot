#!/usr/bin/env python3
"""Export an Ultralytics YOLOv8 detection model to ONNX for the C++ pipeline.

Training / conversion stays in Python; inference is C++ (OnnxDetector).

    pip install ultralytics
    python scripts/export_yolov8.py --weights yolov8n.pt --imgsz 640

Writes models/<name>.onnx.  Run the detector with:

    ./build/apps/stereo_depth_demo --detector onnx --model models/yolov8n.onnx
    ./build/apps/benchmark_detect  --detector onnx --model models/yolov8n.onnx

The exported graph has one input `images` [1,3,imgsz,imgsz] and one output
[1, 84, N] (YOLOv8 layout) which OnnxDetector auto-detects.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="yolov8n.pt", help="Ultralytics weights (auto-downloads)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("error: pip install ultralytics", file=sys.stderr)
        return 1

    model = YOLO(args.weights)
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / pathlib.Path(args.weights).with_suffix(".onnx").name
    shutil.move(str(exported), str(dst))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
