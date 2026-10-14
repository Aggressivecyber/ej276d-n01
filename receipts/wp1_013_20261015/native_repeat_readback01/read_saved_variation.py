"""Only previously saved moments and source lineage from failed attempt2.

No raw SOURCE/REAL container, checkpoint, Torch/model/ADC/AD/optimizer or SSH.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
COLLECTED=ROOT/'runs/wp1_013_20261015/failed_attempt_02'
INVENTORY=ROOT/'receipts/wp1_013_20261015/native_import_fix01/collection_failed_02/REMOTE_ARCHIVE_RECEIPT.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve()
    if out.drive.upper() not in ('E:','D:') or out.exists():raise ValueError('Fresh E/D output required')
    out.mkdir(parents=True)
    inventory=json.loads(INVENTORY.read_text('utf8'))['inventory'];inputs={};members={}
    def bound(name):
        p=COLLECTED/name;r=inventory[name]
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];inputs[name]=r['sha256'];return p
    import numpy as np
    def decode(name,keys):
        path=bound(name)
        with np.load(path,allow_pickle=False) as archive:result={k:archive[k] for k in keys}
        members[name]=keys;return result
    real_meta=json.loads(bound('runs/primary/artifacts/FAILED.json').read_text('utf8'))
    stats={};LSB=1/16383.;r=4;shape=(6593,384)
    for arm in ('A','theta'):
        name='runs/primary/artifacts/SIGNAL_MOMENTS_0_'+arm+'.npz';a=decode(name,['mean_V','sample_variance_V2','source_parent_row'])
        mean,var=a['mean_V'],a['sample_variance_V2']
        assert mean.shape==var.shape==shape and mean.dtype==var.dtype==np.float64 and np.isfinite(mean).all() and np.isfinite(var).all() and (var>=0).all()
        assert np.array_equal(a['source_parent_row'],np.arange(6593))
        per_cell=np.sqrt(var*(r-1)/r);per_parent=np.sqrt(np.mean(var*(r-1)/r,axis=1));rms=float(np.sqrt(np.mean(var)*(r-1)/r))
        largest=np.unravel_index(int(np.argmax(var)),shape);SSE=(r-1)*float(var[largest])
        prefix=float(np.max(np.abs(mean[:,:100])));assert prefix==0 and not np.any(var[:,:100])
        # Var was centered around the saved NumPy mean. All values below retain
        # that numerical reduction; no absent per-draw maxima are invented.
        stats[arm]=dict(parent_count=6593,draws=4,shape=[6593,384],source_prefix_exact_zero=True,
            positive_variance_cells=int((var>0).sum()),positive_variance_parents=int((var>0).any(1).sum()),
            sample_variance_max_V2=float(var.max()),sample_variance_mean_V2=float(var.mean()),
            centered_RMS_V=rms,centered_RMS_ADC_LSB=rms/LSB,
            centered_RMS_over_active_284samples_V=float(np.sqrt(var[:,100:].mean()*(r-1)/r)),
            maximum_cell_centered_RMS_V=float(per_cell.max()),maximum_cell_centered_RMS_ADC_LSB=float(per_cell.max()/LSB),
            individual_repeat_absolute_deviation_from_saved_mean_upper_bound_V=float(np.sqrt(SSE)),
            individual_repeat_absolute_deviation_from_saved_mean_upper_bound_ADC_LSB=float(np.sqrt(SSE)/LSB),
            maximum_pairwise_repeat_difference_upper_bound_V=float(np.sqrt(2*SSE)),
            maximum_pairwise_repeat_difference_upper_bound_ADC_LSB=float(np.sqrt(2*SSE)/LSB),
            upper_bounds_are_not_observed_exact_maxima=True,actual_per_draw_signal_absmax_difference='unknown_not_saved',
            maximum_variance_location=dict(source_parent_row=int(largest[0]),sample=int(largest[1])),
            per_parent_RMS_quantiles_V={str(q):float(np.quantile(per_parent,q)) for q in (0,.5,.9,.99,1)},
            mean_signal_absmax_V=float(np.abs(mean).max()),
            reduction_rounding_note='saved variance includes finite-precision centering; positive tiny variance alone does not prove unequal raw repeats',
            physics_branch='frozen deterministic A; no phys/local randomness' if arm=='A' else 'new shape head with explicit random z')
    lineage=[]
    for bank in (0,1):
        name='runs/primary/artifacts/BANK_'+str(bank)+'.npz';a=decode(name,['source_parent_rows'])['source_parent_rows']
        assert a.shape==(26372,) and a.dtype==np.int64 and np.array_equal(a,np.repeat(np.arange(6593),4))
        lineage.append(dict(bank=bank,rows=len(a),parents=6593,parent_major_four_repeats_exact=True,chunk_rows=128,
            complete_chunks=206,last_chunk_rows=4,no_parent_spans_chunk_boundary=True))
    full_identity=[]
    names=['root/repeated_parent/parent_rows','root/repeated_parent/bank_grid','root/repeated_parent/draw_grid','root/source_input_trends/source_ids']
    original_ids=None
    for arm in ('A','theta'):
        name='runs/primary/evaluation/FULL_0_'+arm+'_fast.npz';a=decode(name,names)
        assert np.array_equal(a[names[0]],np.arange(6593)) and np.all(a[names[1]]==0)
        assert np.array_equal(a[names[2]],np.tile(np.arange(4),(6593,1))) and len(np.unique(a[names[3]]))==6593
        if original_ids is None:original_ids=a[names[3]]
        else:assert np.array_equal(original_ids,a[names[3]])
        full_identity.append(dict(arm=arm,correct_sorted_parent_rows=True,draw_grid=[0,1,2,3],bank_grid=0,unique_source_ids=6593))
    present=sorted(inventory);missing=[name for name in ('runs/primary/artifacts/REFERENCE_0.npz','runs/primary/artifacts/REFERENCE_1.npz',
        'runs/primary/artifacts/SIGNAL_MOMENTS_1_A.npz','runs/primary/artifacts/FULL_POPULATION_COMPARISON.json') if name not in present]
    v=dict(schema='n01/wp1-repeat-saved-readback/v1',status='READBACK_COMPLETE_not_scientific_acceptance',
        inventory_sha256=sha(INVENTORY),input_sha256=inputs,decoded_members=members,
        failed_error=real_meta['error'],actual_completed_counters=real_meta['counters'],inflight=real_meta['inflight'],
        A=stats['A'],theta=stats['theta'],A_centered_RMS_ratio_to_theta=stats['A']['centered_RMS_V']/stats['theta']['centered_RMS_V'],
        lineage=lineage,full_identity=full_identity,missing_raw_evidence=missing,
        saved_A_moments_cannot_identify_exact_max_or_specific_CUDA_kernel=True,cross_bank_A_raw_equivalence='unknown_not_saved',
        all_source_or_REAL_data_container_decodes=0,checkpoint_loads=0,Torch_imported=False,model_forward_rows=0,AD_queries=0,training_updates=0,SSH_calls=0)
    (out/'SAVED_REPEAT_READBACK.json').write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for name,digest in inputs.items():assert sha(COLLECTED/name)==digest
    print(json.dumps(dict(path=str(out/'SAVED_REPEAT_READBACK.json'),A=stats['A'],theta_RMS_V=stats['theta']['centered_RMS_V'],missing=missing),allow_nan=False))

if __name__=='__main__':main()
