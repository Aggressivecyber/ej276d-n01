"""Frozen-membership loss and observation algebra for the bounded012 query.

No data/checkpoint reads, model construction, random sampling, optimizer or
autograd query is performed here. Callers own the single registered graph and
operation accounting. All original numeric loss/ADC sources remain unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from n01.losses008 import _qpsd, _waveform, _wave_views, _extended_log_charge
from n01.losses008_v2 import empirical_wasserstein_1d, WAVE_WEIGHTS, VIEWS, COORDINATE_VIEWS, SHORT_LAGS, _pair_projection
from n01.model import ADC_MIN, ADC_MAX, ADC_LEVELS
from n01.model008 import canonical_charge008
from n01.observation008 import crop_indices_torch, peak_indices_torch


def require(flag, message):
    if not flag:
        raise ValueError(message)


@dataclass(frozen=True)
class FrozenMembership012:
    fake_bins: torch.Tensor
    real_bins: torch.Tensor
    supported: tuple
    counts: tuple
    versions: tuple

    @classmethod
    def from_reference(cls, loss, fake, real):
        with torch.no_grad():
            f = _qpsd(fake.detach(), loss.q_reference, loss.q_floor)
            r = _qpsd(real.detach(), loss.q_reference, loss.q_floor)
            fb = torch.bucketize(f[:, 0].contiguous(), loss.charge_bin_boundaries).detach().clone()
            rb = torch.bucketize(r[:, 0].contiguous(), loss.charge_bin_boundaries).detach().clone()
            counts = tuple((int((fb == i).sum()), int((rb == i).sum())) for i in range(8))
            require(loss.minimum_support == 16, "012 retains original support16")
            supported = tuple(a >= 16 and b >= 16 for a, b in counts)
            return cls(fb, rb, supported, counts, (fb._version, rb._version))

    def validate(self, fake, real):
        require((self.fake_bins._version, self.real_bins._version) == self.versions, "Frozen Q membership was mutated")
        require(len(fake) == len(self.fake_bins) and len(real) == len(self.real_bins), "Frozen membership row count mismatch")
        require(fake.device == self.fake_bins.device and real.device == self.real_bins.device, "Frozen membership device mismatch")
        return self

    def metadata(self):
        return dict(conditional_bins=[dict(bin=i, fake_count=a, real_count=b, supported=self.supported[i])
            for i, (a, b) in enumerate(self.counts)], conditional_bins_used=sum(self.supported),
            frozen_reference="theta2000_actual_hard_observation", minimum_support=16)


def frozen_qpsd(loss, fake, real, context, membership):
    """Exactly original QPSD reduction, with reference masks/support fixed."""
    context.validate()
    membership.validate(fake, real)
    f = loss._z("qpsd", _qpsd(fake, loss.q_reference, loss.q_floor))
    r = loss._z("qpsd", _qpsd(real, loss.q_reference, loss.q_floor))
    directions = context.banks["qpsd"].to(f)
    global_term = .5*(empirical_wasserstein_1d(f, r)+empirical_wasserstein_1d(f@directions, r@directions))
    terms = []
    for i in range(8):
        if membership.supported[i]:
            fm, rm = membership.fake_bins == i, membership.real_bins == i
            terms.append(empirical_wasserstein_1d(f[fm, 1], r[rm, 1]))
    return .5*(global_term+torch.stack(terms).mean()) if terms else global_term


def wave_value(loss, fake, real, context):
    parts = loss.wave_components(fake, real, None, context, mean_response=False)
    return sum(WAVE_WEIGHTS[key]*value for key, value in parts.items()), parts


def native_qpsd(loss, fake, real, context):
    return loss._qpsd_with_context(fake, real, None, context, mean_response=False)


def gather_observation(adc, peaks, q_floor, *, input_carrier=None):
    """Baseline/crop algebra only; never applies ADC (even for validation)."""
    require(adc.ndim == 2 and adc.shape[1] == 384 and bool(torch.isfinite(adc).all()), "012 finite Bx384 ADC values required")
    require(peaks.shape == (len(adc),) and peaks.device == adc.device, "012 crop peak shape/device mismatch")
    indices = crop_indices_torch(peaks)
    prefix = adc[:, :100]
    baseline = prefix.mean(1)
    centered = adc-baseline[:, None]
    window = centered.gather(1, indices)
    q, tail = window.sum(1), window[:, 46:].sum(1)
    nonzero = q != 0
    raw = torch.where(nonzero, tail/torch.where(nonzero, q, torch.ones_like(q)), torch.full_like(q, float("nan")))
    guarded = q.abs() < float(q_floor)
    denominator = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))*q.abs().clamp_min(float(q_floor))
    global_peak = centered.argmax(1)
    flags = dict(peak_at_search_boundary=(peaks == 112) | (peaks == 143),
        global_peak_outside_search=(global_peak < 112) | (global_peak >= 144), Q_nonpositive=q <= 0,
        PSD_undefined=~nonzero, PSD_guarded=guarded, ADC_rail_hit=((adc == ADC_MIN) | (adc == ADC_MAX)).any(1))
    if input_carrier is not None:
        flags["pre_ADC_out_of_range"] = ((input_carrier < ADC_MIN) | (input_carrier > ADC_MAX)).any(1)
    return dict(window_V=window, baseline100=prefix, baseline_mean_V=baseline, adc_carrier=adc,
        peak_index=peaks, global_peak_index=global_peak, crop_indices=indices,
        crop_start=peaks-16, crop_stop=peaks+240, Q=q, tail=tail, PSD_raw=raw,
        PSD=tail/denominator, flags=flags)


def analytic_window_pullback(endpoint_gradient, peaks, input_carrier):
    """Exact fixed-index STE Jacobian, including prefix baseline and rails.

Torch clamp uses the inclusive [ADC_MIN,ADC_MAX] mask at its endpoint values;
those nonsmooth rail cases are separately recorded, not called unique classical
derivatives. The partial gradient of S's prefix can be nonzero even when the
registered head emits identically zero prefix voltage.
"""
    require(endpoint_gradient.shape == (len(input_carrier), 256), "012 endpoint gradient shape mismatch")
    indices = crop_indices_torch(peaks.detach())
    result = torch.zeros_like(input_carrier)
    result.scatter_add_(1, indices, endpoint_gradient.detach())
    result[:, :100] -= endpoint_gradient.detach().sum(1, keepdim=True)/100
    active = (input_carrier.detach() >= ADC_MIN) & (input_carrier.detach() <= ADC_MAX)
    return result*active


def peak_diagnostics(adc):
    search = adc.detach()[:, 112:144]
    ordered = search.sort(1, descending=True).values
    maxima = ordered[:, 0]
    ties = (search == maxima[:, None]).sum(1)
    return dict(first_runnerup_margin_V=maxima-ordered[:, 1], peak_tie_count=ties,
        peak_tied=ties > 1, exact_rail_cell_count=((adc.detach() == ADC_MIN) | (adc.detach() == ADC_MAX)).sum(1))


def source_q_strata(source_carrier384, loss):
    """Original A centering/canonical sum and native frozen loss Q boundaries."""
    x = source_carrier384.detach().double()
    q = canonical_charge008(x-x[:, :100].mean(1, keepdim=True))
    coordinate = _extended_log_charge(q, loss.q_reference, loss.q_floor)
    bins = torch.bucketize(coordinate.contiguous(), loss.charge_bin_boundaries)
    require(bool(torch.isfinite(q).all()), "012 nonfinite canonical SOURCE Q")
    return q, coordinate, bins


def numeric_flags(observation, input_carrier, loss):
    wave = observation["window_V"].detach()
    peak = wave.amax(1)
    q = observation["Q"].detach()
    clipped = input_carrier.detach().clamp(ADC_MIN, ADC_MAX)
    grid = (clipped-ADC_MIN)*ADC_LEVELS
    fraction = grid-grid.floor()
    return dict(**observation["flags"], shape_peak_tied=(wave == peak[:, None]).sum(1) > 1,
        shape_peak_floor_active=peak < loss.peak_floor, shape_peak_floor_equal=peak == loss.peak_floor,
        Q_exact_zero=q == 0, Q_negative=q < 0, Q_log_floor_active=q < loss.q_floor,
        Q_abs_floor_equal=q.abs() == loss.q_floor,
        ADC_halfgrid_tied=(fraction == .5).any(1), ADC_halfgrid_tie_cells=(fraction == .5).sum(1),
        ADC_halfgrid_min_distance=(fraction-.5).abs().amin(1))


@torch.no_grad()
def sorting_signature(loss, wave, context, membership):
    """Indices/tie flags only; retain no extra AD graph or new projection bank."""
    views = _wave_views(wave.double(), loss.peak_floor)
    z = {name: loss._z(name, views[name]) for name in VIEWS}
    matrices = []
    for name in VIEWS:
        matrices.append(("swd:"+name, z[name]@context.banks["wave:"+name].to(z[name])))
    for name in COORDINATE_VIEWS:
        matrices.append(("coordinate:"+name, z[name]))
        for lag in SHORT_LAGS:
            matrices.append((f"lag:{name}:{lag}", _pair_projection(z[name], lag, context.banks[f"lag:{name}:{lag}"])))
    q = loss._z("qpsd", _qpsd(wave, loss.q_reference, loss.q_floor))
    matrices.extend((("q_coordinate", q), ("q_projected", q@context.banks["qpsd"].to(q))))
    for index, supported in enumerate(membership.supported):
        if supported:
            matrices.append(("q_bin:"+str(index), q[membership.fake_bins == index, 1:2]))
    result = {}
    for name, matrix in matrices:
        values, order = matrix.sort(dim=0)
        result[name] = dict(order=order.to(torch.int32), tied_columns=(values[1:] == values[:-1]).any(0),
            tie_adjacent_cells=int((values[1:] == values[:-1]).sum()))
    return result


def sorting_changes(reference, current):
    require(set(reference) == set(current), "012 sorting diagnostic groups changed")
    return {name: dict(permutation_changed_cells=int((row["order"] != current[name]["order"]).sum()),
        permutation_cells=row["order"].numel(), reference_tied_columns=int(row["tied_columns"].sum()),
        probe_tied_columns=int(current[name]["tied_columns"].sum()), reference_tie_adjacent_cells=row["tie_adjacent_cells"],
        probe_tie_adjacent_cells=current[name]["tie_adjacent_cells"],
        interpretation="ordering diagnostic; tied supports do not define unique derivatives") for name, row in reference.items()}


def q_decomposition(signal, noise, observation):
    adc_delta = observation["adc_carrier"]-(signal+noise)
    result = {}
    for name, term in (("signal", signal), ("noise", noise), ("ADC_delta", adc_delta)):
        crop = term.gather(1, observation["crop_indices"])-term[:, :100].mean(1, keepdim=True)
        result["Q_"+name] = crop.sum(1)
        result["tail_"+name] = crop[:, 46:].sum(1)
    result["Q_identity_residual"] = observation["Q"]-sum(result["Q_"+name] for name in ("signal", "noise", "ADC_delta"))
    result["tail_identity_residual"] = observation["tail"]-sum(result["tail_"+name] for name in ("signal", "noise", "ADC_delta"))
    require(float(result["Q_identity_residual"].detach().abs().max()) <= 1e-10
        and float(result["tail_identity_residual"].detach().abs().max()) <= 1e-10, "012 same-crop Q/tail identity failed")
    return result


def flatten_parameters(values):
    return torch.cat([value.detach().reshape(-1).double() for value in values])


def cosine(a, b):
    an, bn = float(a.double().norm()), float(b.double().norm())
    if an == 0 or bn == 0:
        return dict(value=None, reason="zero_l2_norm", norms=[an, bn])
    return dict(value=float((a.double()@b.double()/(an*bn)).clamp(-1, 1)), reason=None, norms=[an, bn])


def gradient_summary(vectors):
    records = {}
    total = vectors["total"]
    norm_total = float(total.norm())
    for name, value in vectors.items():
        require(value.shape == (1288,) and bool(torch.isfinite(value).all()), "012 requires finite full1288 gradient vectors")
        weight, bias = value[:1280].reshape(8, 32, 5), value[1280:]
        records[name] = dict(L2=float(value.norm()), weight_L2=float(weight.norm()), bias_L2=float(bias.norm()),
            basis_slice_L2=[float(torch.cat((weight[k].reshape(-1), bias[k:k+1])).norm()) for k in range(8)],
            cosine_with_total=cosine(value, total),
            signed_projection_on_total=None if norm_total == 0 else float(value@total/norm_total),
            zero_projection_reason="zero_total_l2" if norm_total == 0 else None)
    components = [vectors[k] for k in ("wave", "QPSD", "RMS", "curvature", "saturation")]
    norm_sum = sum(float(value.norm()) for value in components)
    summed = sum(components)
    error = float((total-summed).norm())
    relative = error/max(norm_sum, torch.finfo(torch.float64).tiny)
    require(relative <= 1e-5, "012 total gradient/component sum inconsistency")
    return dict(vectors=records, pairwise_cosines={a: {b: cosine(va, vb) for b, vb in vectors.items()} for a, va in vectors.items()},
        cancellation=dict(sum_component_norms=norm_sum, norm_of_vector_sum=float(summed.norm()),
            retained_norm_ratio=None if norm_sum == 0 else float(summed.norm())/norm_sum,
            cancellation_fraction=None if norm_sum == 0 else 1-float(summed.norm())/norm_sum),
        total_sum_absolute_error=error, total_sum_relative_error=relative, engineering_tolerance=1e-5,
        hypothetical_clip5_coefficient=1. if norm_total <= 5 else 5/norm_total, clipping_applied=False)
