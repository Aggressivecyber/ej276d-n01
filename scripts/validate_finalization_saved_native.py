"""Recompute frozen metrics from saved native outputs; no response inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

import run_response039 as r
from run_response036 import NativeMetricLoss036
from n01.inference028 import verified_path028
from n01.losses014 import fit_integral_statistics014
from n01.response008 import loss_tensor_statistics

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(p,out,device,counts):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    r.write_json(out/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        run_id=p['run_id'],config_sha256=p['_config_sha256'],resources=p['resources']))
    source=Path(p['saved_remote_root'])
    for name,digest in p['saved_file_sha256'].items():
        if r.sha256(source/name)!=digest:raise ValueError('Saved scientific input changed: '+name)
    setup=json.loads((ROOT/p['setup038']['path']).read_text())
    paths={name:verified_path028(setup['inputs'][name]) for name in ('statistics','dataset','tolerances')}
    statistics=json.loads(paths['statistics'].read_text())
    with np.load(paths['dataset']) as archive:
        real=archive['real'].copy();baseline_real=archive['real_baseline100'].copy();roles=archive['real_role'].copy()
    counts['TRAIN_archives_decoded']+=1;counts['REAL_TRAIN_members_decoded']+=3
    if real.shape!=(21047,256) or set(np.unique(roles))!={'TRAIN'}:raise ValueError('Original REAL TRAIN population changed')
    integral=fit_integral_statistics014(torch.as_tensor(real,dtype=torch.float64))
    ev=setup['evaluation']
    loss=NativeMetricLoss036(loss_tensor_statistics(statistics['loss']),integral,
        seed=ev['projection_training_seed'],evaluation_seed=ev['projection_evaluation_seed'],integral_seed=ev['integral_seed']).to(device)
    loss.column_chunk=ev['W1_column_chunk'];loss.last_diagnostics={};context=loss.create_evaluation_context()
    if context.metadata()['fingerprint']!=ev['projection_fingerprint']:raise ValueError('Metric projection changed')
    T=json.loads(paths['tolerances'].read_text())['T_by_hypothesis']
    projection=r.projection039(p['joint_projection_fit_seed'],device)
    real_joint={h:torch.as_tensor(r.joint_embedding(real,baseline_real,np.full(len(real),16),
        dict(T[h],q_floor=float(loss.q_floor))),device=device)@projection for h in r.HYPOTHESES}
    checks=[]
    for head in (0,1):
        for case in ('M0','BASE054','070','S'):
            label=f'seed{head}_{case}'
            storage=json.loads((source/(label+'_STORAGE.json')).read_text())
            for h in r.HYPOTHESES:
                with np.load(source/(label+'_'+h+'_NATIVE.npz')) as data:
                    if not np.array_equal(data['parent_row'],np.arange(6593)):raise ValueError('Saved parent population changed')
                    # Independently decode the complete storage representation.
                    canonical=data['ADC_codes'].astype(np.float64)/16383-.5
                    adc=(canonical.view(np.uint64)^data['ADC_float64_xor']).view(np.float64)
                    peaks=data['peak_index'];centered=adc-data['baseline_mean_V'][:,None]
                    approximate=np.take_along_axis(centered,peaks[:,None]-16+np.arange(256),axis=1)
                    wave=(approximate.view(np.uint64)^data['wave_float64_xor']).view(np.float64)
                    if hashlib.sha256(wave.tobytes()).hexdigest()!=storage[h]['wave_sha256']:raise ValueError('Saved waveform bytes differ')
                    if hashlib.sha256(adc.tobytes()).hexdigest()!=storage[h]['ADC_sha256']:raise ValueError('Saved ADC bytes differ')
                    if not np.array_equal(centered[:,112:144].argmax(1)+112,peaks):raise ValueError('Saved peak differs')
                    flags={name[:-5]:dict(count=int(data[name].sum()),denominator=6593) for name in data.files if name.endswith('_flag')}
                counts['saved_native_archives_decoded']+=1;counts['saved_wave_rows']+=6593
                report=r.hard_metrics(wave,adc[:,:100],real,baseline_real,loss,context)
                z=r.joint_embedding(wave,adc[:,:100],peaks,dict(T[h],q_floor=float(loss.q_floor)))
                report['values']['joint_morphology']=float(r.column_W1(torch.as_tensor(z,device=device)@projection,real_joint[h],16))
                report['values']['negative_lobe_area_Vns']=r.raw_distance(np.maximum(-wave,0).sum(1),np.maximum(-real,0).sum(1))['value']
                report['observation_flags']=flags
                target=out/(label+'_'+h+'_RECOMPUTED.json');r.write_json(target,report)
                expected=json.loads((source/(label+'_'+h+'_METRICS.json')).read_text())
                if json.loads(target.read_text())!=expected:raise RuntimeError('Complete saved-wave metric replay differs: '+label+'/'+h)
                counts['full_metric_reports_verified']+=1
                checks.append(dict(head=head,case=case,hypothesis=h,rows=6593,complete_metric_JSON_exact=True,
                    waveform_bytes_exact=True,ADC_bytes_exact=True,peak_indices_exact=True,
                    preADC_flag_scope='original saved flags retained; preADC carrier is not regenerated'))
                r.check_resources(p,out,started)
                r.write_json(out/'PROGRESS.json',dict(phase='saved_metric_replay',head=head,case=case,hypothesis=h,
                    elapsed_seconds=time.monotonic()-started,execution_counts=dict(counts)))
    if dict(counts)!=p['operation_counts_exact']:raise RuntimeError('Saved replay accounting differs')
    result=dict(status='PASS_FULL_SAVED_NATIVE_METRIC_REPLAY',UTC=datetime.now(timezone.utc).isoformat(),
        run_id=p['run_id'],checks=checks,execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,
        new_response_candidates=0,S_negative_outcome_unchanged=True,scientific_acceptance=False,
        independence='New process starts from saved native codes/XOR/baseline plus original REAL TRAIN and frozen metric definitions; no A0/M0/005 weights, response random draws or response forward. Metric projections reuse their frozen seeds.')
    r.write_json(out/'RESULT.json',result);r.check_resources(p,out,started)
    r.write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,
        result_sha256=r.sha256(out/'RESULT.json'),elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256:raise ValueError('Saved replay config changed')
    p=json.loads(args.config.read_text());p['_config_sha256']=args.config_sha256
    for name,digest in p['source_pins'].items():
        if r.sha256(ROOT/name)!=digest:raise ValueError('Saved replay source changed: '+name)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True);device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    counts=Counter(dict.fromkeys(p['operation_counts_exact'],0))
    try:run(p,args.out,device,counts)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(counts),automatic_restart=False))


if __name__=='__main__':main()
