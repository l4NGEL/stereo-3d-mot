"""Loading one sequence's labels + dumps into a ready-to-score bundle."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

import kitti_io as K
import mot_eval as M


@dataclass
class Bundle:
    seq: str
    n_frames: int
    meta: dict
    labels: pd.DataFrame
    gt: M.SequenceGT
    raw_dir: str
    _dets: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def dets(self) -> pd.DataFrame:
        if self._dets is None:
            self._dets = K.read_dets(os.path.join(self.raw_dir, "dets.txt"))
        return self._dets

    def tracks(self, variant: str) -> pd.DataFrame:
        return K.read_tracks(os.path.join(self.raw_dir, f"tracks_{variant}.txt"))

    def variants(self) -> list[str]:
        return sorted(n[len("tracks_"):-len(".txt")] for n in os.listdir(self.raw_dir)
                      if n.startswith("tracks_") and n.endswith(".txt"))

    def timing(self) -> pd.DataFrame:
        return K.read_timing(os.path.join(self.raw_dir, "timing.csv"))


def load_bundle(raw_root: str, kitti_root: str, seq: str, rules: bool = True) -> Bundle:
    raw_dir = os.path.join(raw_root, seq)
    meta = K.read_meta(os.path.join(raw_dir, "meta.txt"))
    n = int(meta["frames"])
    labels = K.read_labels(os.path.join(kitti_root, "label_02", f"{seq}.txt"))
    return Bundle(seq=seq, n_frames=n, meta=meta, labels=labels,
                  gt=M.SequenceGT(labels, n, rules=rules), raw_dir=raw_dir)
