"""Observable ADC peak structure; no preADC truth or source-event matching."""
import numpy as np

ADC_STEP = 1. / 16383.
PEAK_OFFSETS = np.array([-4, -3, -2, -1, 1, 2, 3, 4])


def peak_code_features(adc, wave, prefix, peaks):
    """Original carrier ADC, original cropped wave and original first argmax.

    Plateaus refer to the maximum ADC code in the fixed [112,144) search.
    No row is dropped, and no generated/noiseless reference is needed.
    """
    codes = np.rint((adc + .5) * 16383.).astype(np.int64)
    search = codes[:, 112:144]
    ordered = np.sort(search, axis=1)
    top = search.max(1)
    is_top = search == top[:, None]
    run = np.zeros(len(codes), dtype=np.int64)
    longest = np.zeros_like(run)
    for j in range(32):
        run = (run + 1) * is_top[:, j]
        longest = np.maximum(longest, run)
    row = np.arange(len(codes))
    selected = codes[row, peaks]
    differences = selected[:, None] - codes[row[:, None], peaks[:, None] + PEAK_OFFSETS]
    centered_prefix = prefix - prefix.mean(1)[:, None]
    variance = np.mean(centered_prefix ** 2, axis=1)
    columns = {
        'Q_Vns': wave.sum(1), 'tail_Vns': wave[:, 46:].sum(1),
        'peak_voltage_V': wave[:, 16],
        'top_two_code_gap_LSB': ordered[:, -1] - ordered[:, -2],
        'maximum_code_tied_samples': is_top.sum(1),
        'maximum_code_longest_plateau': longest,
        'peak_curvature_LSB': 2 * selected - codes[row, peaks - 1] - codes[row, peaks + 1],
        'baseline_mean_V': prefix.mean(1),
        'baseline_RMS_V': np.sqrt(variance),
    }
    for j, offset in enumerate(PEAK_OFFSETS):
        columns[f'peak_minus_neighbor_{offset:+d}_LSB'] = differences[:, j]
    for lag in (1, 2, 5, 10, 20):
        # Raw autocovariance remains defined when a constant baseline has zero RMS.
        columns[f'baseline_ACOV_lag{lag}_V2'] = np.mean(
            centered_prefix[:, lag:] * centered_prefix[:, :-lag], axis=1)
    flags = {
        'search_edge': (peaks == 112) | (peaks == 143),
        'code_tie_at_maximum': is_top.sum(1) > 1,
        'plateau_length_at_least_2': longest >= 2,
        'plateau_length_at_least_3': longest >= 3,
        'Q_nonpositive': columns['Q_Vns'] <= 0,
        'baseline_zero_RMS': variance == 0,
    }
    return columns, flags


def aggregate_code_features(columns, flags):
    levels = [0., .01, .05, .25, .5, .75, .95, .99, 1.]
    summary = {}
    for name, value in columns.items():
        summary[name] = dict(mean=float(value.mean()), std_population=float(value.std()),
                            quantiles=np.quantile(value, levels).tolist(), defined_rows=len(value))
    return dict(rows=len(next(iter(columns.values()))), quantile_levels=levels, features=summary,
                flags={key: dict(count=int(value.sum()), denominator=len(value), rate=float(value.mean()))
                       for key, value in flags.items()}, rows_removed=0,
                peak_structure_scope='same observable ADC search [112,144); no noiseless peak reference')
