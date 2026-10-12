"""Bind a finite zero-training integration study before new scientific reads."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ID='integration_006_20261012'
REC=ROOT/'receipts'/ID

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')

def main():
    if (REC/'PREREGISTRATION.json').exists():raise FileExistsError('Registration already frozen')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head=='dfa06dee5f8fc09b0b0e2c1c74d52e652d43efc4'
    program=read(ROOT/'PROGRAM.json');jobs=read(ROOT/'jobs.json')
    assert program['active_scientific_jobs']==jobs['active_scientific_jobs']==0
    previous=read(ROOT/'NOISE_ACF005_SOURCE_MANIFEST.json')
    for name,digest in previous['files'].items():assert sha(ROOT/name)==digest,name
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    snapshot=ROOT/'provenance/before_integration006';snapshot.mkdir(exist_ok=False)
    files=[]
    for p in (ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT/'README.md',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):
        target=snapshot/p.name;shutil.copy2(p,target)
        files.append(dict(source=str(p),copy=target.relative_to(ROOT).as_posix(),sha256=sha(target)))
    write(snapshot/'SNAPSHOT.json',dict(UTC=now,files=files))
    old=ROOT/'runs/run_001_20261012/final_review_20261012_0900/train'
    paths=dict(noise005=ROOT/'runs/noise_acf005_20261012/runs/science/noise/final.pt',
        full=old/'full/stage_20000.pt',deterministic=old/'deterministic/stage_20000.pt',
        statistics001=old/'fit_statistics.json',complete005=ROOT/'runs/noise_acf005_20261012/runs/science/COMPLETE.json',
        source005=ROOT/'NOISE_ACF005_SOURCE_MANIFEST.json',config005=ROOT/'noise_acf005_config.json',config001=ROOT/'run_config.json',
        final005=ROOT/'receipts/noise_acf005_20261012/FINAL_RECEIPT.json',
        scope005=ROOT/'reports/noise_acf005_20261012/SUPPORT_SCOPE.json',
        original_endpoints=ROOT/'receipts/run_001_20261012/FINAL_EVALUATION_RECEIPT.json')
    folder=ROOT/'provenance/integration006_reference';folder.mkdir(exist_ok=False)
    refs={}
    for key,p in paths.items():
        target=folder/(key+p.suffix);shutil.copy2(p,target)
        assert sha(p)==sha(target)
        refs[key]=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),source=p.relative_to(ROOT).as_posix())
    assert refs['noise005']['sha256']=='1e45d3369b85d2f5f635ad44dbeb5a730f6d86864cc3dc9163384fccd48bd417'
    assert refs['full']['sha256']=='374e33bc4784259cd9315623ba107d81f71f26eeee509a3d517bb2fca7c0c959'
    assert refs['deterministic']['sha256']=='5d632013f106ca7362695d0f083ba302d1232cef35798da1df07f6ff12e518f7'
    prior=read(ROOT/'noise_acf005_config.json')
    cfg=dict(run_id=ID,registered_UTC=now,reviewed_commit=head,
        local_training_container=prior['local_training_container'],remote_training_container=prior['remote_training_container'],
        train_container_sha256=prior['train_container_sha256'],identities_sha256=prior['identities_sha256'],
        remote_root='/home/rylan/projects/ej276d-n01/'+ID,references=refs,
        source_manifest='INTEGRATION006_SOURCE_MANIFEST.json',
        hypotheses=[dict(name='fast',short_weight=1.,short_length=4.,long_weight=0.,long_length=None,long_type='none'),
            dict(name='slow',short_weight=.75,short_length=4.,long_weight=.25,long_length=80.,long_type='exponential'),
            dict(name='common_level',short_weight=.75,short_length=4.,long_weight=.25,long_length=None,long_type='constant')],
        extension=dict(length=480,prefix=100,tail=380,dtype='float64',kernel_ridge_relative=1e-6,
            kernel_variance='mean_diagonal_frozen_C005',mean_reference='mean_frozen_mu005',
            mean_tail='mu0+A*(mu100-mu0)',scale_rule='shared_exp(eta*u-eta^2)',
            fixed_cholesky_only=True,pre_adc_is_effective_latent=True,identified_hardware_noise=False),
        evaluation=dict(arms=['deterministic','full'],splits=['TRAIN','DEV'],draws_per_parent=4,chunk_rows=128,
            train_source_parents=6593,train_real_rows=21047,dev_source_parents=512,dev_real_rows=599,
            original_count_weights=True,old_population_weights_applied=False,
            views=['signal_only','legacy','fast','slow','common_level'],
            sample_waveform_rows=128,noise_diagnostic_rows=8192,
            charge_bins_reference='statistics001.loss.charge_bin_boundaries',minimum_parents_per_fixed_bin=32,
            fixed_bins_reference='preADC_signal_Q',
            minimum_rows_per_fixed_bin=128,bootstrap_draws=1000,bootstrap_cluster='SOURCE_parent_paired_across_hypotheses',
            conditional_width='SD_ddof0_and_IQR_linear_quantile_no_GMM',
            same_treated_REAL='canonical_ADC_then_identical_deterministic_readout_no_random_addition'),
        seeds=dict(model_constructor=2026101260,TRAIN_response=2026101261,TRAIN_noise005=2026101262,
            TRAIN_extension=2026101263,DEV_response=2026101264,DEV_noise005=2026101265,
            DEV_extension=2026101266,noise_diagnostic005=2026101267,noise_diagnostic_extension=2026101268,
            bootstrap_TRAIN=2026101269,bootstrap_DEV=2026101270),
        decision=dict(TRAIN_only=True,Q_mean_range_over_REAL_sd_max=.05,PSD_mean_range_over_REAL_sd_max=.05,
            fixed_bin_PSD_sd_ratio_max=1.10,noise_tail_variance_ratio_max=1.25,
            signal_baseline_absolute_max_V=1/16383,all_fixed_bins_require_support=True,
            bootstrap_interval=[.025,.975],point_thresholds_are_practical=True,
            new_nonpositive_Q_or_guarded_PSD_blocks=True,invalid_Q_scope='TRAIN_three_new_hypotheses_both_arms',
            invalid_Q_outcome='interface_fail',legacy_invalid_Q_exemption=False,
            outcomes=['interface_fail','sensitivity_detected','MC_inconclusive','sensitivity_within_family'],
            interval_rule='any_lower_above_or_analytic_fail_detected;all_upper_within_and_support_within;else_inconclusive',
            response_protocol_must_be_separate=True,scientific_acceptance=False),
        noise_updates=0,response_updates=0,classifier_updates=0,GMM_fits=0,
        cpu_threads=12,gpu_memory_limit_GiB=10,job_tree_RSS_limit_GiB=8,wall_limit_s=1800,output_limit_MiB=256,
        one_active_scientific_job=True,independent_acquisition_available=False,
        validation_previously_exposed=True,no_new_hyperparameter_selection=True)
    write(ROOT/'integration006_config.json',cfg)
    write(REC/'PREREGISTRATION.json',dict(run_id=ID,UTC=now,config_sha256=sha(ROOT/'integration006_config.json'),
        protocol_sha256=sha(ROOT/'docs/INTEGRATION_006_PROTOCOL.md'),references=refs,reviewed_commit=head,
        before_new_scientific_array_decode=True,before_new_scientific_forward=True,
        noise_updates=0,response_updates=0,classifier_updates=0,GMM_fits=0,legacy_results_modified=False))
    program.update(status='integration006_registered',active_repair=ID,active_scientific_jobs=0,
        updated_UTC=now,current_stage='zero_training480_integration_implementation',current_run_training_updates=0,
        integration006_configuration='integration006_config.json',integration006_protocol='docs/INTEGRATION_006_PROTOCOL.md',
        integration006_noise_updates=0,integration006_response_updates=0,
        next_action='Execute fixed zero-training 480 integration sensitivity; decide separately registered two-arm response protocol.')
    write(ROOT/'PROGRAM.json',program)
    jobs.update(updated_UTC=now)
    jobs['jobs'].append(dict(job_id='N01_INTEGRATION006_20261012',status='registered',registered_UTC=now,
        host='100.82.231.102',pid=None,supervisor_pid=None,noise_updates_planned=0,response_updates_planned=0,
        classifier_updates_planned=0,GMM_fits_planned=0,configuration='integration006_config.json',
        protocol='docs/INTEGRATION_006_PROTOCOL.md',preregistration=f'receipts/{ID}/PREREGISTRATION.json',
        remote_root=cfg['remote_root']))
    write(ROOT/'jobs.json',jobs)
    pointer=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';value=read(pointer)
    value.update(status=program['status'],active_repair=ID,active_scientific_jobs=0,updated_UTC=now,
        integration006_configuration=str(ROOT/'integration006_config.json'),
        integration006_protocol=str(ROOT/'docs/INTEGRATION_006_PROTOCOL.md'),integration006_noise_updates=0,
        integration006_response_updates=0)
    write(pointer,value)
    print(json.dumps(dict(status='registered',run_id=ID,config_sha256=sha(ROOT/'integration006_config.json'),
        protocol_sha256=sha(ROOT/'docs/INTEGRATION_006_PROTOCOL.md'),scientific_forward_rows=0)))

if __name__=='__main__':main()
