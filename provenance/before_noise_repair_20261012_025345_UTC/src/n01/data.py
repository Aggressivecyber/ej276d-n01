"""N01 data interface. Only the named training fields are opened for fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path: str | Path, obj) -> None:
    def convert(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=convert) + "\n"
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(text.encode("utf8"))
    tmp.replace(p)


def charge_np(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a[:, 100:400].sum(1) - 3 * a[:, :100].sum(1)


def canonical_real(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if not np.isfinite(a).all() or ((a < -.5) | (a > .5)).any():
        raise ValueError("REAL input is nonfinite/outside recorded ADC interval; no row deletion")
    return np.rint((a + .5) * 16383) / 16383 - .5


def load_training(container: str | Path, identity_path: str | Path) -> dict:
    with np.load(container, allow_pickle=False) as z:
        source = z["source"].copy()
        real = canonical_real(z["target"])
        source_ids = z["source_ids"].copy()
        h = z["h"].astype(np.float64)
    with np.load(identity_path, allow_pickle=False) as z:
        real_ids = z["real_train_original_index"].copy()
        real_groups = z["real_train_group"].copy()
        val_groups = z["real_val_group"].copy()
    if source.shape != (6593, 480) or real.shape != (21047, 480):
        raise ValueError("unexpected original TRAIN shapes")
    if len(np.unique(source_ids)) != len(source) or len(np.unique(real_ids)) != len(real):
        raise ValueError("nonunique parent identities")
    if np.intersect1d(real_groups, val_groups).size:
        raise ValueError("REAL training-validation group overlap")
    for name, v in [("SOURCE", source), ("REAL", real)]:
        if not np.isfinite(v).all() or (charge_np(v) <= 1e-8).any():
            raise ValueError(f"invalid {name} TRAIN waveforms; retained data require diagnosis")
        peak = (v - v[:, :100].mean(1, keepdims=True)).max(1)
        if (peak <= 1e-8).any():
            raise ValueError(f"invalid {name} amplitude")
    if h.shape != (6593,) or not np.isfinite(h).all() or ((h < 0) | (h > 1)).any():
        raise ValueError("invalid original source deposition fraction")
    return dict(source=source, real=real, source_ids=source_ids, real_ids=real_ids,
                h=h, real_groups=real_groups,
                original_source_weights=np.ones(len(source), dtype=np.float64))


def fit_model_statistics(data: dict) -> dict:
    source = data["source"].astype(np.float64)
    real = data["real"].astype(np.float64)
    q = charge_np(source)
    peak = (source - source[:, :100].mean(1, keepdims=True)).max(1)
    baseline = real[:, :100]
    offset = baseline.mean(1)
    centered = baseline - offset[:, None]
    # b1 uses the same complete-window [-1,1] coordinate as ElectronicsNoise.
    t = 2 * np.arange(100, dtype=np.float64) / 479 - 1
    tc = t - t.mean()
    slope = centered @ tc / (tc @ tc)
    midpoint_offset = offset - slope * t.mean()
    rms = (centered - slope[:, None] * tc).std(1, ddof=1)
    qref = float(np.median(q))
    return dict(q_ref=qref, a_ref=float(np.median(peak)),
                log_q_min=float(np.log(q.min())), log_q_max=float(np.log(q.max())),
                electronics_rms=float(rms.mean()),
                electronics_offset_mean=float(midpoint_offset.mean()),
                electronics_offset_std=float(max(midpoint_offset.std(ddof=1), 1e-7)),
                electronics_log_rms_std=float(max(np.log(np.maximum(rms, 1e-8)).std(ddof=1), .01)),
                electronics_slope_std=float(max(slope.std(ddof=1), 1e-7)),
                amplitude_floor=1e-8, charge_floor=1e-8, alignment_t0=120.)


def load_validation(container: str | Path, identity_path: str | Path) -> dict:
    with np.load(container, allow_pickle=False) as z:
        source = z["source_val"].copy()
        real = canonical_real(z["target_val"])
        ids = z["source_val_ids"].copy()
        h = z["hval"].astype(np.float64)
    with np.load(identity_path, allow_pickle=False) as z:
        real_ids = z["real_val_original_index"].copy()
        groups = z["real_val_group"].copy()
    if source.shape != (512, 480) or real.shape != (599, 480):
        raise ValueError("existing validation subset shape changed")
    return dict(source=source, real=real, source_ids=ids, real_ids=real_ids,
                real_groups=groups, h=h)


class FullCoverageSampler:
    """Without replacement within each epoch; carry the final rows into batches."""
    def __init__(self, n: int, seed: int):
        self.n = int(n)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.n)
        self.position = 0
        self.completed_epochs = 0
        self.visits = np.zeros(self.n, dtype=np.int64)

    def draw(self, count: int) -> np.ndarray:
        parts = []
        remaining = int(count)
        while remaining:
            take = min(remaining, self.n - self.position)
            rows = self.order[self.position:self.position + take]
            parts.append(rows)
            np.add.at(self.visits, rows, 1)
            self.position += take
            remaining -= take
            if self.position == self.n:
                self.completed_epochs += 1
                self.order = self.rng.permutation(self.n)
                self.position = 0
        return np.concatenate(parts)

    def state_dict(self) -> dict:
        return dict(n=self.n, rng=self.rng.bit_generator.state,
                    order=torch.from_numpy(self.order.copy()), position=self.position,
                    completed_epochs=self.completed_epochs,
                    visits=torch.from_numpy(self.visits.copy()))

    def load_state_dict(self, state: dict) -> None:
        if state["n"] != self.n:
            raise ValueError("sampler population changed")
        self.rng.bit_generator.state = state["rng"]
        self.order = state["order"].cpu().numpy().copy()
        self.position = state["position"]
        self.completed_epochs = state["completed_epochs"]
        self.visits = state["visits"].cpu().numpy().copy()
