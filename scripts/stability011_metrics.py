"""Exact saved-population metrics; NumPy only, no models/data loading/RNG.

The CDF caches sort all original support once. Repeated cluster occurrences are
integer masses on that support; this is exactly literal row replication, with
separate domain and conditional-bin normalization. No per-parent distances or
quantile approximation replace the pooled waveform/QPSD distances.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

VIEWS = ("voltage", "shape", "difference", "curvature", "difference_smooth_5", "difference_smooth_17")
DIMENSIONS = (256, 256, 255, 254, 255, 255)
COORDINATE_VIEWS = ("voltage", "shape")
LAGS = (1, 2, 4, 8)
WAVE_COLUMNS = 6*32+2*256+2*sum((256-lag)*4 for lag in LAGS)


def require(flag, message):
    if not flag:
        raise ValueError(message)


def counts(value, length):
    value = np.asarray(value, dtype=np.float64)
    require(value.shape == (length,), "Row-multiplicity length mismatch")
    require(np.isfinite(value).all() and (value >= 0).all() and (value == np.floor(value)).all()
            and value.sum() > 0, "Multiplicities must be finite nonnegative integers with positive mass")
    return value


@dataclass(frozen=True)
class CDFCache:
    order: np.ndarray
    gap: np.ndarray
    fake_rows: int
    real_rows: int

    @classmethod
    def create(cls, fake, real):
        f, r = np.asarray(fake, np.float64), np.asarray(real, np.float64)
        if f.ndim == 1:
            f = f[:, None]
        if r.ndim == 1:
            r = r[:, None]
        require(f.ndim == r.ndim == 2 and f.shape[1] == r.shape[1] and min(len(f), len(r)) > 0,
                "CDF requires nonempty matrices with equal column count")
        require(np.isfinite(f).all() and np.isfinite(r).all(), "Nonfinite CDF support")
        support = np.concatenate((f.T, r.T), axis=1)
        order = np.argsort(support, axis=1, kind="stable")
        gap = np.ascontiguousarray(np.diff(np.take_along_axis(support, order, axis=1), axis=1))
        dtype = np.uint16 if support.shape[1] <= 65536 else np.uint32
        order = np.ascontiguousarray(order, dtype=dtype)
        order.flags.writeable = gap.flags.writeable = False
        return cls(order, gap, len(f), len(r))

    @property
    def bytes(self):
        return self.order.nbytes+self.gap.nbytes

    def distance(self, fake_counts, real_counts, *, column_chunk=256):
        fw, rw = counts(fake_counts, self.fake_rows), counts(real_counts, self.real_rows)
        require(isinstance(column_chunk, int) and column_chunk > 0, "Positive CDF column chunk required")
        mass = np.concatenate((fw/fw.sum(), -rw/rw.sum()))
        total = 0.
        for start in range(0, len(self.order), column_chunk):
            stop = min(start+column_chunk, len(self.order))
            cumulative = mass[self.order[start:stop]]
            np.cumsum(cumulative, axis=1, out=cumulative)
            work = cumulative[:, :-1]
            np.abs(work, out=work)
            np.multiply(work, self.gap[start:stop], out=work)
            total += float(work.sum(dtype=np.float64))
        result = total/len(self.order)
        require(np.isfinite(result) and result >= 0, "Nonfinite/negative empirical W1")
        return result


def wave_views(wave, peak_floor):
    x = np.asarray(wave, np.float64)
    require(x.ndim == 2 and x.shape[1] == 256 and len(x) > 0 and np.isfinite(x).all(), "Expected finite N x 256 waveform")
    shape = x/np.maximum(x.max(axis=1, keepdims=True), float(peak_floor))
    result = dict(voltage=x, shape=shape, difference=np.diff(shape, axis=1), curvature=np.diff(shape, n=2, axis=1))
    for size in (5, 17):
        padded = np.pad(shape, ((0, 0), (size//2, size//2)), mode="edge")
        smooth = np.lib.stride_tricks.sliding_window_view(padded, size, axis=1).mean(axis=2)
        result["difference_smooth_"+str(size)] = np.diff(smooth, axis=1)
    return result


def q_features(wave, stats):
    x = np.asarray(wave, np.float64)
    q = x.sum(axis=1, dtype=np.float64)
    floor, reference = float(stats["q_floor"]), float(stats["q_reference"])
    require(floor > 0 and reference > 0, "Positive frozen Q resolution/reference required")
    logq = np.where(q >= floor, np.log(np.maximum(q, floor)/reference),
        np.log(floor/reference)+np.arcsinh((q-floor)/floor))
    denominator = np.where(q < 0, -1., 1.)*np.maximum(np.abs(q), floor)
    raw = np.column_stack((logq, x[:, 46:256].sum(axis=1)/denominator))
    require(np.isfinite(raw).all(), "Nonfinite guarded QPSD feature")
    return raw, q


def standard(value, item):
    center, scale = np.asarray(item["center"], np.float64), np.asarray(item["scale"], np.float64)
    require(np.isfinite(center).all() and np.isfinite(scale).all() and (scale > 0).all(), "Invalid frozen standardization")
    return (value-center)/scale


@dataclass(frozen=True)
class Representation:
    groups: dict
    q_coordinate: np.ndarray
    q_projected: np.ndarray
    q_bins: np.ndarray
    nonpositive: np.ndarray
    guarded: np.ndarray
    rows: int

    @classmethod
    def create(cls, wave, stats, banks):
        values = wave_views(wave, stats["peak_floor"])
        require(tuple(values) == VIEWS, "Frozen six-view order changed")
        standardized = {name: standard(values[name], stats["views"][name]) for name in VIEWS}
        groups = {"swd": [], "coordinate": [], "short_lag": []}
        for name, dimension in zip(VIEWS, DIMENSIONS):
            direction = np.asarray(banks["wave:"+name], np.float64)
            require(direction.shape == (dimension, 32), "Frozen wave direction shape changed")
            groups["swd"].append(standardized[name] @ direction)
        for name in COORDINATE_VIEWS:
            value = standardized[name]
            groups["coordinate"].append(value)
            for lag in LAGS:
                direction = np.asarray(banks[f"lag:{name}:{lag}"], np.float64)
                require(direction.shape == (2, 4), "Frozen short-lag direction shape changed")
                groups["short_lag"].append((value[:, :-lag, None]*direction[0]
                    +value[:, lag:, None]*direction[1]).reshape(len(value), -1))
        raw, q = q_features(wave, stats)
        qz = standard(raw, stats["qpsd"])
        direction = np.asarray(banks["qpsd"], np.float64)
        require(direction.shape == (2, 16), "Frozen QPSD direction shape changed")
        edges = np.asarray(stats["charge_bin_boundaries"], np.float64)
        require(edges.shape == (7,) and np.isfinite(edges).all() and (np.diff(edges) >= 0).all(), "Invalid frozen Q boundaries")
        bins = np.searchsorted(edges, raw[:, 0], side="left")
        return cls(groups, qz, qz @ direction, bins, q <= 0, np.abs(q) < stats["q_floor"], len(q))


@dataclass(frozen=True)
class WholeProxyCache:
    groups: dict
    global_q: tuple
    conditional: tuple
    fake: Representation
    real: Representation
    minimum_support: int

    @classmethod
    def create(cls, fake, real, minimum_support=16):
        groups = {name: tuple(CDFCache.create(f, r) for f, r in zip(fake.groups[name], real.groups[name]))
            for name in ("swd", "coordinate", "short_lag")}
        global_q = (CDFCache.create(fake.q_coordinate, real.q_coordinate), CDFCache.create(fake.q_projected, real.q_projected))
        conditional = []
        for index in range(8):
            fm, rm = fake.q_bins == index, real.q_bins == index
            value = CDFCache.create(fake.q_coordinate[fm, 1], real.q_coordinate[rm, 1]) if fm.any() and rm.any() else None
            conditional.append((fm, rm, value))
        # Retain only tiny row metadata and conditional coordinates after caching.
        fmeta = Representation({}, fake.q_coordinate, np.empty((0, 0)), fake.q_bins, fake.nonpositive, fake.guarded, fake.rows)
        rmeta = Representation({}, real.q_coordinate, np.empty((0, 0)), real.q_bins, real.nonpositive, real.guarded, real.rows)
        require(minimum_support == 16, "Preserve original minimum conditional support16")
        return cls(groups, global_q, tuple(conditional), fmeta, rmeta, minimum_support)

    @property
    def bytes(self):
        values = [c for group in self.groups.values() for c in group]+list(self.global_q)
        values += [c for _, _, c in self.conditional if c is not None]
        return sum(c.bytes for c in values)

    def evaluate(self, fake_counts, real_counts, *, column_chunk=256):
        fw, rw = counts(fake_counts, self.fake.rows), counts(real_counts, self.real.rows)
        parts = {name: float(np.mean([c.distance(fw, rw, column_chunk=column_chunk) for c in self.groups[name]]))
            for name in ("swd", "coordinate", "short_lag")}
        global_term = .5*sum(c.distance(fw, rw, column_chunk=column_chunk) for c in self.global_q)
        diagnostics, terms = [], []
        for index, (fm, rm, cached) in enumerate(self.conditional):
            nf, nr = int(fw[fm].sum()), int(rw[rm].sum())
            supported = nf >= self.minimum_support and nr >= self.minimum_support
            diagnostics.append(dict(bin=index, fake_count=nf, real_count=nr, supported=supported))
            if supported:
                terms.append(cached.distance(fw[fm], rw[rm], column_chunk=column_chunk))
        conditional_mean = float(np.mean(terms)) if terms else None
        qpsd = .5*(global_term+conditional_mean) if terms else global_term
        wave = .5*parts["swd"]+.25*parts["coordinate"]+.25*parts["short_lag"]
        require(np.isfinite([wave, qpsd, global_term, *parts.values()]).all(), "Nonfinite whole proxy")
        return dict(components=dict(wave=wave, QPSD=qpsd), wave_subcomponents=parts,
            proxy=wave+2*qpsd, QPSD_global=global_term, QPSD_conditional_mean=conditional_mean,
            conditional_counts=diagnostics, conditional_bins_used=len(terms), conditional_coverage_complete=len(terms) == 8,
            no_supported_bin_behavior="global_term_only_with_incomplete_coverage_diagnostic",
            fake_rows=int(fw.sum()), real_rows=int(rw.sum()),
            fake_nonpositive_q_rows=int(fw[self.fake.nonpositive].sum()), real_nonpositive_q_rows=int(rw[self.real.nonpositive].sum()),
            fake_guarded_psd_rows=int(fw[self.fake.guarded].sum()), real_guarded_psd_rows=int(rw[self.real.guarded].sum()))


def paired_record(metrics):
    a, final = metrics
    sa = float(np.mean([value["proxy"] for value in a]))
    sf = float(np.mean([value["proxy"] for value in final]))
    require(sa > 0 and np.isfinite([sa, sf]).all(), "Positive finite A proxy required for relative contrast")
    return dict(S_A=sa, S_final=sf, Delta=sf-sa, relative_contrast_percent=100*(sf/sa-1),
        conditional_coverage_complete=all(value["conditional_coverage_complete"] for arm in metrics for value in arm))
