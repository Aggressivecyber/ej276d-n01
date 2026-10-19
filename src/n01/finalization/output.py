"""Save and reopen a complete research inference result without pickle."""
import hashlib
import json
from pathlib import Path

import numpy as np


def save_result(output_dir, result, provenance):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    names = ('window_V','baseline100','baseline_mean_V','peak_index','global_peak_index',
             'crop_start','crop_stop','crop_indices','carrier_adc','pre_adc_carrier',
             'Q','front','tail','PSD_raw','PSD','row_parent','draw')
    arrays = {name:result[name].detach().cpu().numpy() for name in names}
    arrays['parent_ids'] = np.asarray(result['parent_ids'])
    groups = ['flags']
    if 'readout_features' in result:
        groups += ['readout_features','readout_flags','diagnostic_flags']
    if 'contributions' in result:
        groups.append('contributions')
    for group in groups:
        arrays.update({group+'__'+name:value.detach().cpu().numpy()
                       for name,value in result[group].items()})
    path = output_dir/'OUTPUT.npz'
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        for name, array in arrays.items():
            if saved[name].dtype != array.dtype or saved[name].shape != array.shape or saved[name].tobytes() != array.tobytes():
                raise RuntimeError('Inference output failed byte-exact reopening: '+name)
    metadata = {key:result[key] for key in ('streams','applicability','model_seed',
                                           'noise_hypothesis','detector_config')}
    if 'QP_parent_status' in result:
        metadata['QP_parent_status'] = result['QP_parent_status']
    record = dict(schema='n01/finalization-inference-output/v1', status='PASS_LOSSLESS_REOPEN',
        rows=len(result['window_V']), parents=len(result['parent_ids']), output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        output_bytes=path.stat().st_size, metadata=metadata, provenance=provenance,
        arrays={name:dict(shape=list(value.shape), dtype=value.dtype.str,
                         sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name,value in arrays.items()})
    (output_dir/'RECEIPT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return record
