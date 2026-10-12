"""One-time prospective registration; no waveform decoding or scientific fits."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
ID = 'noise_origin_audit_003_20261012'
OLD = ROOT.parent / 'sim2real_two_papers_20261008_0305/work/R14_TRAIN_POOL_EXPANSION_001'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')

def main():
    receipt = ROOT / 'receipts' / ID / 'PREREGISTRATION.json'
    if receipt.exists():
        raise FileExistsError('Registration already exists; preserve prospective evidence')
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    snapshot = ROOT / 'provenance' / 'before_noise_origin_003'
    snapshot.mkdir(parents=True, exist_ok=False)
    copies = [ROOT/'PROGRAM.json', ROOT/'jobs.json', ROOT/'README.md',
              ROOT/'repair_run_config.json', ROOT/'.gitattributes', ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json']
    manifest = []
    for p in copies:
        dest = snapshot/p.name
        shutil.copy2(p, dest)
        manifest.append({'source':str(p), 'copy':str(dest.relative_to(ROOT)), 'sha256':sha(dest)})
    write(snapshot/'SNAPSHOT.json', {'UTC':now, 'files':manifest})
    reference = ROOT/'provenance/noise_origin_003_reference'
    reference.mkdir(parents=True,exist_ok=False)
    for name in ('interface_core.py','extract_train.py'):
        shutil.copy2(OLD/name,reference/name)
    raw = json.loads((OLD/'config.json').read_text(encoding='utf8'))['remote_source']
    raw['io_caps']['hash'] = raw['container']['bytes']
    raw['io_caps']['source_total'] += raw['container']['bytes']
    refs = json.loads((ROOT/'DATA_REFERENCES.json').read_text(encoding='utf8'))
    config = {
        'experiment_id':ID, 'registered_UTC':now,
        'training_container':{'path':refs['training_container']['path'], 'sha256':refs['training_container']['container_sha256']},
        'identities':{'path':'data/PARENT_IDENTITIES.npz','sha256':refs['real_parent_identity_binding']['sha256']},
        'raw_source':raw,
        'reference_interface':{'path':'provenance/noise_origin_003_reference/interface_core.py','sha256':sha(reference/'interface_core.py')},
        'origin_audit':{'expected_train_rows':21047,'expected_train_groups':422,'baseline_samples':100,'raw_baseline_samples':2000,'fixed_window_starts':[0,5,10,15,20],'bootstrap_resamples':512,'bootstrap_seed':2026101259,'save_block_means':True,'baseline_positions_are_pre_pulse_not_proven_noise':True},
        'rr_calibration':{'repeats':128,'classifier_updates':200,'classifier_fit_per_class':2048,'classifier_test_per_class':256,'bootstrap_resamples':1000,'test_label_permutations':255,'pseudo_test_groups':12,'min_group_rows':44,'pseudo_split_seed':2026101260,'split_seed_base':2026101261,'classifier_seed_base':2026102261,'permutation_seed_base':2026103261,'bootstrap_seed_base':2026104261,'view':'raw','significance_alpha':0.05},
        'resource':{'cpu_threads':12,'rss_GiB':8,'gpu_memory_GiB':10,'wall_limit_s':1800,'output_limit_GiB':1},
        'training_updates':{'noise':0,'response':0,'evaluation_classifier_planned':25600},
        'validation_waveform_decode_allowed':False,'modify_old_gate_allowed':False,
        'remote_root':'/home/rylan/projects/ej276d-n01/noise_origin_audit_003_20261012',
        'remote_training_container':'/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R33_REMOTE_20261009_175239_UTC/inputs/TRAIN_PRIVATE.npz',
        'scientific_interpretation':'conditional_origin_diagnostic_and_null_calibration_not_independent_acquisition_validation',
    }
    write(ROOT/'noise_origin_003_config.json', config)
    write(receipt, {'experiment_id':ID,'UTC':now,'config_sha256':sha(ROOT/'noise_origin_003_config.json'),'protocol_sha256':sha(ROOT/'docs/NOISE_ORIGIN_003_PROTOCOL.md'),'before_any_new_raw_decode_or_classifier_fit':True,'reference_interface_sha256':sha(reference/'interface_core.py'),'noise_updates':0,'response_updates':0,'classifier_update_budget':25600,'old_noise_gate_modified':False})
    program = json.loads((ROOT/'PROGRAM.json').read_text(encoding='utf8'))
    program.update(status='noise_origin_audit_registered', current_stage='origin_audit_implementation_and_review', active_audit=ID, active_scientific_jobs=0, updated_UTC=now, origin_audit_config='noise_origin_003_config.json', origin_audit_protocol='docs/NOISE_ORIGIN_003_PROTOCOL.md', origin_noise_training_updates=0, origin_response_training_updates=0, next_action='Execute prospective TRAIN-only pipeline reconstruction and fixed-budget REAL-REAL calibration; no noise or response training.')
    write(ROOT/'PROGRAM.json',program)
    jobs=json.loads((ROOT/'jobs.json').read_text(encoding='utf8'))
    jobs['updated_UTC']=now
    jobs['jobs'].append({'job_id':'N01_NOISE_ORIGIN_AUDIT_003_20261012','status':'registered','registered_UTC':now,'host':'100.82.231.102','pid':None,'noise_updates':0,'response_updates':0,'classifier_updates_planned':25600,'configuration':'noise_origin_003_config.json','protocol':'docs/NOISE_ORIGIN_003_PROTOCOL.md','preregistration':str(receipt.relative_to(ROOT)),'remote_root':config['remote_root']})
    write(ROOT/'jobs.json',jobs)
    pointerp=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'
    pointer=json.loads(pointerp.read_text(encoding='utf8'))
    pointer.update(status=program['status'],updated_UTC=now,active_audit=ID,active_scientific_jobs=0,origin_audit_config=str(ROOT/'noise_origin_003_config.json'))
    write(pointerp,pointer)
    print(json.dumps({'registered':ID,'UTC':now,'receipt':str(receipt),'config_sha256':sha(ROOT/'noise_origin_003_config.json')},ensure_ascii=False))

if __name__=='__main__':
    main()
