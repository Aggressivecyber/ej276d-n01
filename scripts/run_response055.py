"""Confirm the exact054 endpoint on unopened banks; no calibration or reselection."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import sys,time
import numpy as np
import torch
import run_response039 as r
from run_response052 import evaluate052

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run055(p,out,device,progress):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    r.write_json(out/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),run_id=p['run_id'],
        config_sha256=p['_config_sha256'],pid=os.getpid(),resources=p['resources'],phase='setup'))
    frozen=json.loads((ROOT/p['frozen_endpoint_file']).read_text())
    if p['endpoint']!=frozen['endpoint']:raise RuntimeError('055 endpoint differs from predeclared054 freeze')
    r.write_json(out/'ENDPOINT_FROZEN.json',dict(UTC=datetime.now(timezone.utc).isoformat(),endpoint=p['endpoint'],
        original054_frozen_sha256=frozen['original054_frozen_sha256'],confirmation_purpose=p['confirmation_purpose'],
        unopened_confirmation_banks_per_head=p['confirmation_banks'],joint_projection_confirmation_seed=p['joint_projection_confirmation_seed'],
        no_endpoint_reselection=True,parameters_frozen_before_any_bank=True))
    setup=r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()),device,progress)
    for model in setup['models']:model.set_phase('frozen').eval()
    T=json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis']
    q=r.canonical_charge008(setup['encoding']['original']['signal_pre']).cpu().numpy()
    if hashlib.sha256(q.tobytes()).hexdigest()!=p['input_Q_array_sha256']:raise RuntimeError('055 full input Q hash differs')
    real=setup['data']['real'];baseline=setup['data']['real_baseline100']
    embedding={h:torch.as_tensor(r.joint_embedding(real,baseline,np.full(len(real),16),
        dict(T[h],q_floor=float(setup['loss'].q_floor))),device=device) for h in r.HYPOTHESES}
    projection=r.projection039(p['joint_projection_fit_seed'],device);real_joint={h:z@projection for h,z in embedding.items()}
    replays=json.loads((ROOT/p['replay_references_file']).read_text())
    for index in (0,1):
        r.check_resources(p,out,started)
        bank=r.make_population_bank(setup,index,'FIT',0,p,progress)
        reference,_=evaluate052(setup,index,np.zeros(8),bank,T,projection,real_joint,p,out,progress)
        base,bgeo=evaluate052(setup,index,p['base_bias'],bank,T,projection,real_joint,p,out,progress,
            restitution_fraction=p['base_restitution_fraction'],geometry_rows=f'seed{index}_FIT_BASE042')
        endpoint,geo=evaluate052(setup,index,p['base_bias'],bank,T,projection,real_joint,p,out,progress,
            restitution_fraction=p['base_restitution_fraction'],step_size=1.,geometry_rows=f'seed{index}_FIT_ENDPOINT')
        path=out/f'seed{index}_FIT_REPLAY.json'
        r.write_json(path,dict(M0=reference,BASE042=base,candidate=endpoint,base_geometry=bgeo,geometry=geo))
        saved=json.loads(path.read_text())
        if any(saved[key]!=replays[str(index)][key] for key in ('M0','BASE042','candidate')):
            raise RuntimeError('055 full054 FIT replay failed')
        progress['fit_replay_heads']+=1
        r.write_json(out/'PROGRESS.json',dict(phase='exact054_FIT_replay',seed_index=index,
            elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
        del bank
    projection=r.projection039(p['joint_projection_confirmation_seed'],device)
    real_joint={h:z@projection for h,z in embedding.items()}
    results=[]
    for index in (0,1):
        pairs=[];base_pairs=[];increments={h:{k:[] for k in ('wave','joint_morphology','morphology_mean_T_gain')} for h in r.HYPOTHESES}
        for bank_id in range(p['confirmation_banks']):
            r.check_resources(p,out,started)
            bank=r.make_population_bank(setup,index,p['confirmation_purpose'],bank_id,p,progress)
            reference,_=evaluate052(setup,index,np.zeros(8),bank,T,projection,real_joint,p,out,progress)
            base,bgeo=evaluate052(setup,index,p['base_bias'],bank,T,projection,real_joint,p,out,progress,
                restitution_fraction=p['base_restitution_fraction'],geometry_rows=f'seed{index}_CONFIRM_{bank_id}_BASE042')
            endpoint,geo=evaluate052(setup,index,p['base_bias'],bank,T,projection,real_joint,p,out,progress,
                restitution_fraction=p['base_restitution_fraction'],step_size=1.,geometry_rows=f'seed{index}_CONFIRM_{bank_id}_ENDPOINT')
            pairs.append((endpoint,reference));base_pairs.append((base,reference))
            for h in r.HYPOTHESES:
                for k in ('wave','joint_morphology'):increments[h][k].append((base[h]['values'][k]-endpoint[h]['values'][k])/base[h]['values'][k])
                increments[h]['morphology_mean_T_gain'].append(float(np.mean([(base[h]['values'][k]-endpoint[h]['values'][k])/T[h][k] for k in r.MORPH])))
            r.write_json(out/f'seed{index}_CONFIRM_{bank_id}.json',dict(seed_index=index,bank_id=bank_id,purpose=p['confirmation_purpose'],
                stream_seeds={k:r.seed039(p['seeds'][index],p['confirmation_purpose'],bank_id,k) for k in ('shape','noise005','extension')},
                M0=reference,BASE042=base,candidate=endpoint,base_geometry=bgeo,geometry=geo,
                endpoint_decision=r.objective039(endpoint,reference,T,p['contract']),BASE042_decision=r.objective039(base,reference,T,p['contract'])))
            progress['confirmation_banks_completed']+=1
            r.write_json(out/'PROGRESS.json',dict(phase='fresh_frozen_endpoint_confirmation',seed_index=index,completed_banks_for_head=bank_id+1,
                elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
            del bank
        decision=r.confirmation039(pairs,T,p['contract'])
        results.append(dict(seed_index=index,decision=decision,BASE042_decision=r.confirmation039(base_pairs,T,p['contract']),
            incremental_gain_vs_BASE042={h:{k:r.estimate039(values) for k,values in row.items()} for h,row in increments.items()},
            incremental_comparison_is_descriptive_not_a_new_acceptance_rule=True))
        r.write_json(out/f'seed{index}_CONFIRMATION_DECISION.json',results[-1])
    r.check_resources(p,out,started)
    for index,model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict())!=r.state_digest(setup['heads'][index]):raise RuntimeError('055 M0 head changed')
    if r.state_digest(setup['baseline'].state_dict())!=setup['frozen_A_digest']:raise RuntimeError('055 A0 changed')
    if r.state_digest(setup['extension'].state_dict())!=setup['frozen005_digest']:raise RuntimeError('055 005 changed')
    if dict(progress)!=p['operation_bounds']:raise RuntimeError('055 completed operation counts differ from contract')
    useful=all(z['decision']['useful_mean_correction'] for z in results)
    result=dict(schema='n01/response055-result/v1',run_id=p['run_id'],endpoint=p['endpoint'],results=results,
        stage_outcome='USEFUL_MEAN_STAGE' if useful else 'CLOSED_NEGATIVE_CONFIRMATION',useful_mean_correction_both_seeds=useful,
        scientific_status='response_match_not_established',full_random_response_acceptance=False,confirmation_banks_opened=8,
        DEV_opened=False,all_TRAIN_SOURCE_parents=6593,all_TRAIN_REAL_rows=21047,all_parent_count_weights_preserved=True,
        execution_counts=dict(progress),elapsed_seconds=time.monotonic()-started,frozen_A0_M0_005_verified=True,hardware_calibrated=False,
        uncertainty_scope='Four fresh CRN contrasts per original head; fixed TRAIN SOURCE and one REAL acquisition; no endpoint reselection',
        next_step='Verify exact replay, fresh bank identities, every original039 guard and signed confirmation failure; continue the necessary evidence-backed response work.')
    r.write_json(out/'RESULT.json',result)
    r.write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,
        result_sha256=r.sha256(out/'RESULT.json'),elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256:raise ValueError('055 config hash mismatch')
    p=json.loads(args.config.read_text())
    for name,digest in p['source_pins'].items():
        if r.sha256(ROOT/name)!=digest:raise ValueError('055 source pin mismatch: '+name)
    if r.sha256(ROOT/p['setup038']['path'])!=p['setup038']['sha256']:raise ValueError('055 setup hash mismatch')
    p['_config_sha256']=args.config_sha256;os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    progress=Counter(dict.fromkeys(p['operation_bounds'],0))
    try:run055(p,args.out,device,progress)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),
            execution_counts=dict(progress),automatic_restart=False,scientific_status='response_match_not_established'))


if __name__=='__main__':main()
