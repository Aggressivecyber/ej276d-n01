"""Prospective registration only; no arrays, model initialization or fitting."""
from pathlib import Path
import copy
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
ID='noise_improvement_004_20261012'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')

def main():
    receipt=ROOT/'receipts'/ID/'PREREGISTRATION.json'
    if receipt.exists(): raise FileExistsError('Preserve registered run')
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    snapshot=ROOT/'provenance/before_noise_improvement_004'
    snapshot.mkdir(parents=True,exist_ok=False)
    files=[]
    for p in [ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT/'README.md',ROOT/'repair_run_config.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json']:
        q=snapshot/p.name;shutil.copy2(p,q)
        files.append(dict(source=str(p),copy=str(q.relative_to(ROOT)),sha256=sha(q)))
    write(snapshot/'SNAPSHOT.json',dict(UTC=now,files=files))
    prior=json.loads((ROOT/'repair_run_config.json').read_text(encoding='utf8'))
    refs=json.loads((ROOT/'DATA_REFERENCES.json').read_text(encoding='utf8'))
    config=dict(run_id=ID,registered_UTC=now,noise=copy.deepcopy(prior['noise']),
        train_container_sha256=prior['train_container_sha256'],identities_sha256=prior['identities_sha256'],
        local_training_container=refs['training_container']['path'],
        remote_training_container='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R33_REMOTE_20261009_175239_UTC/inputs/TRAIN_PRIVATE.npz',
        remote_root='/home/rylan/projects/ej276d-n01/'+ID,
        model=dict(type='ObservedBaselineNoiseV3',length=100,covariance_ridge_relative=1e-6,
            normalized_diagonal_floor=1e-6,initial_log_scale_std=0.02,scale_second_moment=1,
            init_covariance_ddof=1,full_mean_and_cholesky_trainable=True),
        cpu_threads=12,gpu_memory_limit_GiB=10,job_tree_RSS_limit_GiB=8,wall_limit_s=1800,output_limit_MiB=512,
        checkpoint_policy='fixed_final_5000_only_no_validation_selection',
        response_updates=0,response_integration_allowed=False,
        old_gates_changed=False,evaluation_seeds_reused_for_comparability=True,
        validation_previously_used_for_diagnostics=True,independent_acquisition_available=False,
        initialization_role='all_legal_REAL_TRAIN_first100_observed_baseline',
        descriptive_covariance_metrics_are_acceptance_gates=False,
        one_active_scientific_job=True)
    write(ROOT/'noise_improvement_004_config.json',config)
    write(receipt,dict(run_id=ID,UTC=now,config_sha256=sha(ROOT/'noise_improvement_004_config.json'),
        protocol_sha256=sha(ROOT/'docs/NOISE_IMPROVEMENT_004_PROTOCOL.md'),before_new_scientific_initialization=True,
        before_new_scientific_array_decode=True,noise_update_budget=5000,classifier_update_budget=1200,
        response_update_budget=0,old_gates_changed=False,prior_v2_config_sha256=sha(ROOT/'repair_run_config.json')))
    p=json.loads((ROOT/'PROGRAM.json').read_text(encoding='utf8'))
    if p['active_scientific_jobs']!=0: raise RuntimeError('Another scientific job is registered active')
    p.update(status='noise_improvement_registered',active_repair=ID,active_scientific_jobs=0,
        current_run_training_updates=0,current_stage='bounded_noise_implementation',updated_UTC=now,
        improvement_configuration='noise_improvement_004_config.json',improvement_protocol='docs/NOISE_IMPROVEMENT_004_PROTOCOL.md',
        improvement_noise_updates=0,improvement_response_updates=0,
        next_action='Fit one bounded observed100 baseline candidate; unchanged gate; no 480-sample or pulse integration.')
    write(ROOT/'PROGRAM.json',p)
    j=json.loads((ROOT/'jobs.json').read_text(encoding='utf8'));j['updated_UTC']=now
    j['jobs'].append(dict(job_id='N01_NOISE_IMPROVEMENT_004_20261012',status='registered',registered_UTC=now,
        host='100.82.231.102',pid=None,noise_updates_planned=5000,classifier_updates_planned=1200,response_updates_planned=0,
        configuration='noise_improvement_004_config.json',protocol='docs/NOISE_IMPROVEMENT_004_PROTOCOL.md',
        preregistration=str(receipt.relative_to(ROOT)),remote_root=config['remote_root']))
    write(ROOT/'jobs.json',j)
    pp=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';ptr=json.loads(pp.read_text(encoding='utf8'))
    ptr.update(status=p['status'],active_repair=ID,active_scientific_jobs=0,updated_UTC=now)
    write(pp,ptr)
    print(json.dumps(dict(registered=ID,UTC=now,config_sha256=sha(ROOT/'noise_improvement_004_config.json'))))
if __name__=='__main__': main()
