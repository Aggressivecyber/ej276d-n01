"""Read-only PSD width summaries of saved paired slope interventions."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from n01.data import write_json, sha256
from n01.evaluate import _fit_mixture, _clean


def main():
    torch.set_num_threads(2)
    frozen = ROOT/'runs/run_001_20261012/final_review_20261012_0900/train'
    paired = ROOT/'runs/noise_audit_20261012/slope/paired_cuda'
    stats = torch.load(frozen/'full/stage_20000.pt', map_location='cpu', weights_only=False)['loss_statistics']
    qref, qfloor = float(stats['q_reference']), float(stats['q_floor'])
    edges = np.asarray(stats['charge_bin_boundaries'])
    def membership(q):
        x = np.where(q >= qfloor, np.log(np.maximum(q,qfloor)/qref), np.log(qfloor/qref)+np.arcsinh((q-qfloor)/qfloor))
        return np.searchsorted(edges,x,side='left')
    def describe(psd, q, mask, parent=None):
        values = psd[mask]
        invalid = ~np.isfinite(values)
        result = dict(rows=len(values),nonfinite_PSD_rows=int(invalid.sum()),nonpositive_Q_rows=int((q[mask]<=0).sum()),
                      PSD_outside_0_1_rows=int(((values<0)|(values>1)).sum()), no_rows_dropped=True,
                      source_distinct_parents=None if parent is None else int(len(np.unique(parent[mask]))))
        if invalid.any() or len(values)<2:
            result['status']='undefined_or_insufficient_retained_population'
        else:
            result.update(mean_PSD=float(values.mean()),std_PSD_ddof1=float(values.std(ddof=1)),
                          quantiles_05_50_95=np.quantile(values,[.05,.5,.95]),mixture=_fit_mixture(values))
        return result
    with np.load(frozen/'full/evaluation/endpoint_populations.npz') as z:
        realq,realpsd=z['REAL_Q'],z['REAL_PSD']
    rb=membership(realq)
    result=dict(scope='Postprocessing only; frozen TRAIN charge bins; descriptive unlabeled components',
        response_training_updates=0,additional_response_generations=0,charge_boundaries_Vns=qref*np.exp(edges),
        REAL=[describe(realpsd,realq,rb==k) for k in range(8)],arms={})
    for arm in ['full','deterministic']:
        path=paired/arm/'paired_outputs.npz'
        with np.load(path) as z:
            onq,offq=z['on_Q'],z['off_Q'];onpsd,offpsd=z['on_PSD_raw'],z['off_PSD_raw']
            ids=z['source_parent_ids'][z['flat_parent_index']]
        onb,offb=membership(onq),membership(offq)
        result['arms'][arm]=dict(paired_file_sha256=sha256(path),rows_migrating_output_charge_bin=int((onb!=offb).sum()),
            on_output_Q_bins=[describe(onpsd,onq,onb==k,ids) for k in range(8)],
            off_output_Q_bins=[describe(offpsd,offq,offb==k,ids) for k in range(8)],
            off_with_on_bin_membership_held=[describe(offpsd,offq,onb==k,ids) for k in range(8)])
    write_json(paired/'CONDITIONAL_PSD_WIDTHS.json',_clean(result))
    print(json.dumps({a:{'migrated':v['rows_migrating_output_charge_bin'],'on_std':[x.get('std_PSD_ddof1') for x in v['on_output_Q_bins']],
                         'off_std':[x.get('std_PSD_ddof1') for x in v['off_output_Q_bins']]} for a,v in result['arms'].items()}))


if __name__=='__main__':main()
