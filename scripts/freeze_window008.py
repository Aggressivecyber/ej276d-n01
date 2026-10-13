"""Freeze the reviewed training sources, native receipts and prepared TRAIN bytes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import datetime as dt

ROOT=Path(__file__).resolve().parents[1]
RUN_ID='window008_20261013'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(path.read_text(encoding='utf8'))


def bound(root,name):
    path=(root/name).resolve()
    require(path.is_relative_to(root.resolve()),'Bound path escapes run root: '+str(name))
    return path


def check_hashes(root,hashes,label,required=()):
    require(bool(hashes) and set(required)<=set(hashes),label+' has incomplete source bindings')
    for name,digest in hashes.items():
        require(sha(bound(root,name))==digest,label+' source differs: '+name)


def write(path,value):
    # Exclusive publication preserves a prior successful or interrupted attempt.
    with path.open('x',encoding='utf8') as stream:
        stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def validate_evidence(root,cfg):
    """Read-only gates shared by freeze and dispatch; never decode waveforms."""
    require(cfg['run_id']==RUN_ID,'Unexpected run ID')
    config_sha=sha(root/'window008_config.json')
    rec=root/'receipts/window008_20261013'
    qa_path=rec/'engineering_cuda_01/ENGINEERING_RECEIPT.json'
    full_path=rec/'engineering_fullbatch_01/BENCHMARK.json'
    native=rec/'engineering_supervisor_01/exit_receipt.json'
    full_native=rec/'supervisor_fullbatch_01/exit_receipt.json'
    steady_path=rec/'engineering_fullbatch_steady_01/BENCHMARK.json'
    steady_native=rec/'supervisor_fullbatch_steady_01/exit_receipt.json'
    core=('src/n01/model008.py','src/n01/response008.py','src/n01/observation008.py',
          'src/n01/losses008.py','src/n01/losses008_v2.py')
    qa=read(qa_path)
    require(qa['status']=='passed_synthetic_engineering_only' and qa['scientific_updates']==0
            and qa['device']=='cuda:0' and qa['config_sha256']==config_sha,'CUDA engineering receipt is not current/passed')
    require(qa['fixed_noise005_sha256']==cfg['noise_reference']['sha256'],'CUDA noise005 binding differs')
    check_hashes(root,qa['source_sha256'],'CUDA engineering',core)
    full=read(full_path)
    require(full['status']=='passed_native_fullbatch_capacity_only' and full['device']=='cuda:0'
            and full['config_sha256']==config_sha,'Full-batch receipt is not current/passed')
    require(full['noise005_sha256']==cfg['noise_reference']['sha256'],'Full-batch noise005 binding differs')
    check_hashes(root,full['source_sha256'],'Full-batch engineering',core+('scripts/benchmark_window008.py',))
    expected=dict(microbatches=cfg['distribution_microbatches'],source_parents_per_microbatch=cfg['source_parents_per_microbatch'],
                  responses_per_parent=cfg['responses_per_parent'],real_rows_per_microbatch=cfg['real_rows_per_microbatch'],
                  distribution_rows_per_hypothesis=cfg['distribution_microbatches']*cfg['source_parents_per_microbatch']*cfg['responses_per_parent'])
    require(full['formal_batch']==expected,'Full-batch dimensions differ')
    require(full['engineering_G_updates']==2 and full['engineering_D_updates_total']==3,'Full-batch optimizer accounting differs')
    for key in ('scientific_G_updates','scientific_D_updates','noise_updates','population_updates','TRAIN_rows_read','DEV_rows_read'):
        require(full[key]==0,'Full-batch receipt records unexpected work: '+key)
    require(full['method_settings_changed'] is False and full['hyperparameters_selected'] is False
            and full['real_or_source_data_decoded'] is False,'Full-batch scope changed')
    require([row['stage'] for row in full['steps']]==['mean_response','joint_stochastic'],'Full-batch stages differ')
    for row in full['steps']:
        require(row['step_completed'] is True and row['engine_metrics']['same_version_replay_verified'] is True,'Full-batch step/replay incomplete')
        require(max(row['gpu_peak_allocated_bytes'],row['gpu_peak_reserved_bytes'])<=cfg['gpu_memory_limit_GiB']*1024**3,'Full-batch GPU cap exceeded')
        require(row['rss_tree_sampled_peak_bytes']<=cfg['job_tree_RSS_limit_GiB']*1024**3,'Full-batch RSS cap exceeded')
    steady=read(steady_path)
    require(steady['status']=='passed_native_fullbatch_steady_engineering_only' and steady['device']=='cuda:0'
            and steady['config_sha256']==config_sha,'Steady full-batch receipt is not current/passed')
    require(steady['formal_batch']==expected and steady['noise005_sha256']==cfg['noise_reference']['sha256'],'Steady batch/noise binding differs')
    check_hashes(root,steady['source_sha256'],'Steady full-batch engineering',core+('scripts/benchmark_window008_steady.py','src/n01/train_window008.py'))
    require(steady['engineering_G_updates']==20 and steady['engineering_D_updates_total']==30,'Steady optimizer accounting differs')
    for key in ('scientific_G_updates','scientific_D_updates','noise_updates','population_updates','TRAIN_rows_read','DEV_rows_read'):
        require(steady[key]==0,'Steady receipt records unexpected work: '+key)
    require(steady['method_settings_changed'] is False and steady['hyperparameters_selected'] is False,'Steady engineering scope changed')
    require([(row['stage'],row['stage_step']) for row in steady['steps']]==[(stage,step) for stage in ('mean_response','joint_stochastic') for step in range(1,11)],'Steady stages/counts differ')
    for row in steady['steps']:
        require(row['step_completed'] is True,'Steady engineering step incomplete')
        require(max(row['gpu_peak_allocated_bytes'],row['gpu_peak_reserved_bytes'])<=cfg['gpu_memory_limit_GiB']*1024**3,'Steady GPU cap exceeded')
        require(row['rss_tree_sampled_peak_bytes']<=cfg['job_tree_RSS_limit_GiB']*1024**3,'Steady RSS cap exceeded')
        require(row['engine_metrics']['same_version_replay_verified']==row['verify_replay'],'Steady replay cadence differs')
    for path in (native,full_native,steady_native):
        receipt=read(path)
        require(receipt['exit_code']==0 and receipt.get('reason') is None,'Native engineering did not exit cleanly: '+str(path))
    review=read(rec/'training_driver_review.json')
    require(review['status']=='passed_static_and_isolated_control_flow_after_fix','Training-driver review not passed')
    check_hashes(root,review['checked_files_sha256'],'Training-driver review',('src/n01/train_window008.py','src/n01/response008.py','window008_config.json','scripts/launch_window008.sh','scripts/supervise.py'))
    losses=read(rec/'loss_review.json')
    require(losses['status']=='passed_after_two_review_fixes','Loss review not passed')
    check_hashes(root,losses['reviewed_code_sha256'],'Loss review',('src/n01/losses008.py','src/n01/losses008_v2.py'))
    require(sha(bound(root,cfg['dataset']))==cfg['dataset_sha256'] and sha(bound(root,cfg['statistics']))==cfg['statistics_sha256'],'Prepared input binding differs')
    require(sha(root/'data/PARENT_IDENTITIES.npz')==cfg['identities_sha256'],'Identity binding differs')
    require(sha(bound(root,cfg['noise_reference']['path']))==cfg['noise_reference']['sha256'],'Frozen005 binding differs')
    data=read(rec/'data_verification.json');runtime=read(rec/'RUNTIME_INPUT_BINDING.json')
    require(data['status']=='passed' and data['output_data_sha256']==cfg['dataset_sha256'] and data['DEV_waveforms_decoded'] is False,'Prepared data review differs')
    require(data['input_sha256']==dict(container=cfg['train_container_sha256'],identities=cfg['identities_sha256']),'Original data bindings differ')
    require(runtime['dataset_sha256']==cfg['dataset_sha256'] and runtime['statistics_sha256']==cfg['statistics_sha256']
            and runtime['config_sha256']==config_sha and runtime['DEV_waveforms_decoded'] is False,'Runtime staging binding differs')
    return dict(qa=qa_path,native=native,full=full_path,full_native=full_native,steady=steady_path,steady_native=steady_native)


def main():
    path=ROOT/'WINDOW008_SOURCE_MANIFEST.json'
    if path.exists():raise FileExistsError(path)
    cfg=read(ROOT/'window008_config.json')
    rec=ROOT/'receipts/window008_20261013'
    evidence=validate_evidence(ROOT,cfg)
    qa_path,native,full_native=evidence['qa'],evidence['native'],evidence['full_native']
    for name in ('runs/science','runs/REMOTE_START.json','runs/START_ATTEMPT.json'):
        require(not (ROOT/name).exists(),'Science dispatch already attempted: '+name)
    require(not (rec/'PREREGISTRATION.json').exists(),'Preserve previous preregistration attempt')
    excluded={'evaluate_window008.py','train_response007_continuation01.py'}
    files=[p for p in (ROOT/'src/n01').glob('*.py') if p.name not in excluded]
    files += [ROOT/name for name in ('AGENTS.md','USER_PROPOSAL.md','window008_config.json',
        'scripts/check_window008_engine.py','scripts/benchmark_window008.py','scripts/benchmark_window008_steady.py','scripts/supervise.py',
        'scripts/launch_window008.sh','scripts/start_window008_remote.py','scripts/freeze_window008.py',
        'scripts/prepare_window008.py','scripts/fit_window008_statistics.py','scripts/prepare_window008_runtime.py',
        'docs/WINDOW008_PROTOCOL.md','docs/WINDOW008_DATA_INTERFACE.md','docs/WINDOW008_MODEL_DESIGN.md','docs/WINDOW008_NOISE_DESIGN.md',
        'data/PARENT_IDENTITIES.npz',cfg['noise_reference']['path'])]
    files += list((ROOT/'data/window008').glob('*'))
    files += [qa_path,native,full_native,rec/'data_verification.json',rec/'loss_review.json',rec/'training_driver_review.json',rec/'RUNTIME_INPUT_BINDING.json']
    files += list((rec/'engineering_fullbatch_01').glob('*.json'))
    files += list((rec/'engineering_fullbatch_steady_01').glob('*.json'))+[evidence['steady_native']]
    registration=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),run_id=cfg['run_id'],
        user_window_request='那就缩短 和我们实际测到数据齐平 我记得是256?',
        user_parallel_authorization='不用核查这些了 规划下一步 并允许使用子智能体也是astramax快速 快速并行推进完成项目',
        final_observation_samples=256,internal_carrier_samples=384,all_TRAIN_source=6593,all_TRAIN_real=21047,
        config_sha256=sha(ROOT/'window008_config.json'),protocol_sha256=sha(ROOT/'docs/WINDOW008_PROTOCOL.md'),
        G_budget=57000,D_budget=144000,noise_updates=0,population_updates=0,DEV_waveforms_used_for_training=False,
        source_proposal='USER_PROPOSAL.md_with_user_window_revision',scientific_status='response_match_not_established',
        engineering_native_exit_code=0,engineering_steps_are_not_scientific_updates=True,
        evaluation_freeze='separate_after_all_frozen_final_checkpoints_no_checkpoint_selection')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    # One immutable claim prevents concurrent freezes or silent retry after a crash.
    write(rec/'FREEZE_ATTEMPT.json',dict(UTC=registration['UTC'],config_sha256=registration['config_sha256'],source_commit=commit))
    write(rec/'PREREGISTRATION.json',registration);files.append(rec/'PREREGISTRATION.json')
    files.append(rec/'FREEZE_ATTEMPT.json')
    result=dict(schema='n01/window008/source_manifest/v1',run_id=cfg['run_id'],UTC=registration['UTC'],source_commit=commit,
        config_sha256=registration['config_sha256'],before_scientific_model_forward=True,
        TRAIN_preparation_already_executed=True,DEV_waveforms_decoded=False,
        engineering_cuda_receipt=qa_path.relative_to(ROOT).as_posix(),engineering_cuda_native_exit=native.relative_to(ROOT).as_posix(),
        full_batch_native_exit=full_native.relative_to(ROOT).as_posix(),
        full_batch_receipt=evidence['full'].relative_to(ROOT).as_posix(),
        steady_full_batch_receipt=evidence['steady'].relative_to(ROOT).as_posix(),
        steady_full_batch_native_exit=evidence['steady_native'].relative_to(ROOT).as_posix(),
        files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))})
    write(path,result)
    print(json.dumps(dict(status='training_source_frozen',source_commit=commit,files=len(result['files']),manifest_sha256=sha(path))))


if __name__=='__main__':main()
