"""Lossless native ADC and observed-wave storage with exact floating residuals."""
import hashlib
import numpy as np


def save_native(path, observation):
    adc = observation['carrier_adc'].cpu().numpy()
    baseline = observation['baseline_mean_V'].cpu().numpy()
    peak = observation['peak_index'].cpu().numpy()
    wave = observation['window_V'].cpu().numpy()
    codes = np.rint((adc+.5)*16383).astype(np.uint16)
    canonical = codes.astype(np.float64)/16383-.5
    adc_xor = adc.view(np.uint64)^canonical.view(np.uint64)
    approximate = np.take_along_axis(adc-baseline[:, None], peak[:, None]-16+np.arange(256), axis=1)
    wave_xor = wave.view(np.uint64)^approximate.view(np.uint64)
    np.savez_compressed(path, parent_row=np.arange(len(adc)), ADC_codes=codes,
        ADC_float64_xor=adc_xor, baseline_mean_V=baseline, peak_index=peak,
        wave_float64_xor=wave_xor,
        **{k+'_flag': v.cpu().numpy() for k, v in observation['flags'].items()})
    with np.load(path) as saved:
        restored_adc, restored_wave, restored_prefix = load_native(saved)
    if (restored_adc.tobytes() != adc.tobytes() or restored_wave.tobytes() != wave.tobytes()
            or restored_prefix.tobytes() != observation['baseline100'].cpu().numpy().tobytes()):
        raise RuntimeError('Native scientific output did not reopen byte-exactly')
    return dict(rows=len(adc), ADC_float64_xor_nonzero=int(np.count_nonzero(adc_xor)),
                wave_float64_xor_nonzero=int(np.count_nonzero(wave_xor)),
                ADC_sha256=hashlib.sha256(adc.tobytes()).hexdigest(),
                wave_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                lossless_reopen_verified=True)


def load_native(saved):
    canonical = saved['ADC_codes'].astype(np.float64)/16383-.5
    adc = (canonical.view(np.uint64)^saved['ADC_float64_xor']).view(np.float64)
    approximate = np.take_along_axis(adc-saved['baseline_mean_V'][:, None],
                                    saved['peak_index'][:, None]-16+np.arange(256), axis=1)
    wave = (approximate.view(np.uint64)^saved['wave_float64_xor']).view(np.float64)
    return adc, wave, adc[:, :100]
