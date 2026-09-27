#!/usr/bin/env python3
"""Machine-readable experiment inventory (PROTOCOL.md §2, §3.7): hashes of the model, configs and
protocol, a manifest digest for every dataset directory used, tool versions and the hardware.
Written to docs/evaluation/results/inventory.json; INVENTORY.md is the human-readable version."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(root: str, exts=(".png", ".txt", ".pfm")) -> dict:
    """Digest of (relative path, sha256) over every file under root, plus counts and bytes."""
    lines, total, n = [], 0, 0
    for base, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            if not name.lower().endswith(exts):
                continue
            p = os.path.join(base, name)
            lines.append(f"{os.path.relpath(p, root).replace(os.sep, '/')} {sha256(p)}")
            total += os.path.getsize(p)
            n += 1
    return {"files": n, "bytes": total, "manifest_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/evaluation/results/inventory.json")
    a = ap.parse_args()
    inv: dict = {"git_head": run(["git", "rev-parse", "HEAD"]), "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"])}

    inv["files"] = {p: sha256(p) for p in (
        "models/yolov8n.onnx", "configs/kitti.yaml", "configs/kitti_tiled.yaml", "configs/default.yaml",
        "configs/eval/variants_dev.txt", "configs/eval/variants_test.txt", "docs/evaluation/PROTOCOL.md",
        "scripts/export_yolov8.py") if os.path.exists(p)}
    inv["model_bytes"] = os.path.getsize("models/yolov8n.onnx")
    try:
        import onnx
        m = onnx.load("models/yolov8n.onnx", load_external_data=False)
        inv["model_onnx"] = {"producer": m.producer_name, "producer_version": m.producer_version,
                             "opset": [o.version for o in m.opset_import],
                             "input": [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
                             "output": [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output],
                             "metadata": {p.key: (p.value[:200]) for p in m.metadata_props}}
    except Exception as e:  # noqa: BLE001
        inv["model_onnx"] = f"unavailable ({e})"

    data = {}
    for name, path in (("kitti_tracking_dev_and_test", "data/kitti"), ("kitti_odometry", "data/kitti_odometry"),
                       ("kitti_stereo2015", "data/kitti_stereo2015")):
        if os.path.isdir(path):
            data[name] = dir_digest(path)
    inv["datasets"] = data

    inv["host"] = {"os": platform.platform(), "python": sys.version.split()[0], "machine": platform.machine(),
                   "cpu": run(["powershell", "-NoProfile", "-Command",
                               "(Get-CimInstance Win32_Processor | Select-Object -First 1 | ForEach-Object { $_.Name + ' / ' + $_.NumberOfCores + 'C ' + $_.NumberOfLogicalProcessors + 'T' })"]),
                   "ram_gb": run(["powershell", "-NoProfile", "-Command", "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"]),
                   "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
                   "docker": run(["docker", "version", "--format", "{{.Server.Version}}"])}
    pk = {}
    for mod in ("numpy", "scipy", "pandas", "motmetrics", "cv2", "onnx", "onnxruntime"):
        try:
            pk[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pk[mod] = "n/a"
    inv["host_python_packages"] = pk
    inv["container"] = run(["docker", "run", "--rm", "stereo-3d-mot:deps-onnx", "bash", "-c",
                            "echo ubuntu=$(. /etc/os-release; echo $VERSION_ID); echo opencv=$(pkg-config --modversion opencv4); "
                            "echo gcc=$(gcc -dumpfullversion); echo cmake=$(cmake --version | head -1); echo ort=$(ls /opt/onnxruntime/lib | grep -o 'libonnxruntime.so.[0-9.]*' | head -1)"]).split("\n")
    inv["image_ids"] = run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.ID}}", "stereo-3d-mot"]).split("\n")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(inv, open(a.out, "w"), indent=1)
    print(json.dumps({k: inv[k] for k in ("git_head", "model_bytes", "container", "host")}, indent=1))
    print({k: (v["files"], round(v["bytes"] / 1e9, 2)) for k, v in data.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
