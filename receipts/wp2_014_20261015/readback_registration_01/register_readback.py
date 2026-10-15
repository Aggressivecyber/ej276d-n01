"""Root registration of one bounded saved-array CPU audit; no remote action."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT = Path('E:/GAN/research/N01')
BASE = ROOT/'receipts/wp2_014_20261015'
OUT = BASE/'readback_registration_01'
SIDE = '/home/rylan/projects/ej276d-n01/wp2_014_saved_readback_20261015'
RUN = 'wp2_014_saved_readback_20261015'

def read(p):
    return json.loads(p.read_bytes())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def record(p):
    return dict(bytes=p.stat().st_size, sha256=sha(p))

def write(p, value, exclusive=False):
    with p.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

native = read(BASE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json')
collection = read(BASE/'collection_01/COLLECTION_RECEIPT.json')
selection = read(BASE/'postrun_engineering_01/DEVELOPMENT_SELECTION.json')
assert native['cumulative_training_updates'] == 167667
assert collection['whole_remote_inventory_verified'] and collection['selected_file_count'] == 142
assert not collection['all_raw_artifacts_collected'] and collection['raw_artifacts_retained_remote']
assert selection['development_path'] == 'C1' and selection['weights_promoted'] is False
assert selection['independent_raw_readback_complete'] is False
worker = BASE/'postrun_independent_01/readback_saved_arrays.py'
assert sha(worker) == 'de071016fc926f8bc8c5cc475c4e4e6a230047166d9f43343deca1554d625caf'
assert sha(BASE/'postrun_independent_01/saved_array_kernels.py') == 'fe1366f35f36e97974f1a4037a36ad858c03dd8453957c1b45b65080f0fee3ac'
assert sha(BASE/'postrun_independent_01/READBACK_PLAN.json') == '2ca90515781e4cc8759c3986e6cdadda6d786b6d887ad2c828fa5d5b2c08b357'
program = read(ROOT/'PROGRAM.json')
assert program['new_training_updates'] == 167667 and program['active_scientific_jobs'] == 0
frozen = read(BASE/'runtime_package_01/FREEZE_RECEIPT.json')
for rel, expected in frozen['source_inventory'].items():
    assert record(ROOT/rel) == expected, rel
inventory = OUT/'BOUND_ORIGINAL_INVENTORY.json'
assert not inventory.exists()
shutil.copy2(BASE/'collection_01/REMOTE_FULL_INVENTORY.json', inventory)
assert sha(inventory) == '4cf27f6c771294561a474ef2de9574ab711d9322c588da488f0d9ce2e49792d5'
paths = {
    'readback_control.py': BASE/'readback_engineering_01/readback_control.py',
    'readback_saved_arrays.py': worker,
    'saved_array_kernels.py': BASE/'postrun_independent_01/saved_array_kernels.py',
    'READBACK_PLAN.json': BASE/'postrun_independent_01/READBACK_PLAN.json',
    'BOUND_ORIGINAL_INVENTORY.json': inventory,
}
now = dt.datetime.now(dt.timezone.utc).isoformat()
config = dict(schema='n01/wp2-independent-readback-execution/v1', run_id=RUN,
    registered_UTC=now, status='registered_not_started', remote_root=SIDE,
    original_root='/home/rylan/projects/ej276d-n01/wp2_014_20261015',
    host='rylan@100.82.231.102', python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python',
    files={name:dict(path=SIDE+'/'+name, local_path=str(p), **record(p)) for name,p in paths.items()},
    resource_limits=dict(CPU_threads=2, RSS_MiB=1536, virtual_memory_MiB=3072, wall_seconds=1800, output_MiB=64),
    selected_collection_MiB_cap_including_controls_and_logs=80,
    original_files_read_only=True, automatic_retry=False,
    exclusive_project_lock='/home/rylan/projects/ej276d-n01/.n01_scientific_dispatch.lock',
    scientific_budget=read(BASE/'postrun_independent_01/READBACK_PLAN.json')['scientific_budget'],
    finite_scope=dict(bank_pairs=36, raw_NPZ=176, hard_rows_NPZ=72, REPORT_NPZ=72,
        baseline100_NPZ=16, probe_raw_NPZ=8, probe_gradient_NPZ=8, SOURCE_parents=6593,
        rows_per_hard_file=26372, waveform_rows_decoded=1898784, raw_SHA_passes=1,
        raw_decode_passes=1, input_compressed_envelope_MiB=6160, input_IO_envelope_MiB=12320),
    root_authority=dict(user_master_plan='USER_MASTER_PLAN_20261015.txt',
        user_master_plan_sha256=sha(ROOT/'USER_MASTER_PLAN_20261015.txt'),
        latest_user_instruction='2026-10-15T02:30:00.398Z n01 heartbeat',
        authorizes_one_bounded_saved_array_audit=True, no_additional_user_permission_required=True),
    original_training_source_commit='e116368c4f699243a60598c03865bcb4f601be35',
    original_configuration_sha256='abb1de68d5ca312d1dbaf14dc058d7514429321b0813aa00c70ff37b20e92e32',
    original_source_manifest_sha256='14db81d3a08a0edede97f544ed8ec4c63147138223c00fc38d833bbc278e2684',
    original_training_completed_G=8000, original_training_completed_AD=9872,
    cumulative_training_before_and_after_this_audit=167667,
    original_raw_NPZ_remain_remote=True, original_raw_local_collection_complete=False,
    new_original_TRAIN_or_DEV_waveform_reads=0,
    raw_complete_wave_QPSD_noise_loss_replay=False,
    loss_evidence='Frozen producer and independent JSON aggregation; saved row features and r4/r8 moments independently recalculated by this audit.',
    C1_median_fit_independently_replayed=False, preADC_fullcarrier_independent_reconstruction=False,
    provisional_development_path='C1', weights_promoted=False,
    long_job_policy='Verify first formal saved-array pair and same-snapshot health; save one future wake and rest without continuous monitoring.',
    scientific_status='response_match_not_established', scientific_acceptance=False)
write(OUT/'READBACK_EXECUTION.json', config, True)
ref = dict(configuration=str((OUT/'READBACK_EXECUTION.json').relative_to(ROOT)),
    configuration_sha256=sha(OUT/'READBACK_EXECUTION.json'), run_id=RUN, remote_root=SIDE,
    resource_limits=config['resource_limits'], source_file_count=5, source_controls=6,
    source_commit_pending=True, status='registered_not_started', new_training_updates=0,
    new_AD_queries=0, NN_forward_rows=0, scientific_status='response_match_not_established')
cref = dict(path=str((BASE/'collection_01/COLLECTION_RECEIPT.json').relative_to(ROOT)),
    sha256=sha(BASE/'collection_01/COLLECTION_RECEIPT.json'), selected_files=142, full_inventory_files=318,
    omitted_raw_files=176, all_raw_collected=False, raw_NPZ_retained_remote=True)
sref = dict(path=str((BASE/'postrun_engineering_01/DEVELOPMENT_SELECTION.json').relative_to(ROOT)),
    sha256=sha(BASE/'postrun_engineering_01/DEVELOPMENT_SELECTION.json'), development_path='C1',
    status='provisional_pending_independent_raw_readback', weights_promoted=False,
    both_candidates_ALL6_wave_usefulness_pass=False, scientific_acceptance=False)
metas = [ROOT/'MASTER_EXECUTION_PLAN.json', ROOT/'PROGRAM.json', ROOT/'jobs.json',
    ROOT/'README.md', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
backup = OUT/'metadata_before_registration'
backup.mkdir()
for p in metas:
    shutil.copy2(p, backup/p.name)
for p in [x for x in metas if x.suffix == '.json']:
    d = read(p)
    d['updated_UTC'] = now
    d['active_scientific_jobs'] = 0
    d['scientific_status'] = 'response_match_not_established'
    if p.name == 'jobs.json':
        job = next(x for x in d['jobs'] if x.get('run_id') == 'wp2_014_20261015')
        job.update(status='completed_native0_selectively_collected_raw_readback_pending',
            collection_complete=True, collection=cref, independent_readback_complete=False,
            provisional_development_selection=sref)
        job['execution']['status'] = job['status']
        assert not any(x.get('run_id') == RUN for x in d['jobs'])
        d['jobs'].append(dict(job_id='N01_WP2_014_SAVED_ARRAY_READBACK_20261015', run_id=RUN,
            scientific_job=True, kind='CPU_saved_array_audit', status='registered_not_started',
            registered_UTC=now, execution=ref, resource_limits=config['resource_limits'],
            G_updates=0, new_AD_queries=0, NN_forward_rows=0, checkpoint_loads=0,
            automatic_retry=False, original_training_before_and_after=167667,
            scientific_status='response_match_not_established'))
    else:
        d.update(status='WP2_014_saved_array_readback_prepared', current_work_package='WP2',
            current_stage='WP2_014_independent_saved_array_readback_prepared', active_research=RUN,
            wp2_collection=cref, wp2_readback_execution=ref, wp2_provisional_development_selection=sref,
            next_action='Start one registered zero-NN saved-array CPU audit; verify a formal pair, set one wake and rest. Then close WP2 and continue bounded WP3.')
        if 'wp2_execution' in d:
            d['wp2_execution'].update(status='completed_native0_selectively_collected_raw_readback_pending',
                collection_complete=True, collection=cref, independent_readback_complete=False)
        if p.name == 'PROGRAM.json':
            d.update(wp2_collection_complete=True, wp2_independent_readback_complete=False,
                active_audit=RUN, new_training_updates=167667)
        if p.name == 'MASTER_EXECUTION_PLAN.json':
            d['active_work_package'] = 'WP2'
            wp = next(x for x in d['work_packages'] if x['id'] == 'WP2')
            wp.update(status='native0_selectively_collected_independent_raw_readback_prepared',
                collection=cref, saved_array_readback=ref, provisional_development_selection=sref)
            wp['execution']['status'] = 'completed_native0_selectively_collected_raw_readback_pending'
            d['current_action'] = d['next_action']
    write(p, d)
readme = ROOT/'README.md'
lead = '''# N01 当前执行：WP2/014 保存数组独立复核准备

014 四次配对短试全部完成，每次 2000G，总计 8000G / 9872AD，原生计算与监督均退出0。累计训练167667，已含启动197G，搜索短试已用4次；WP3保留原定最多6次。

保存指标按冻结规则选择 C1 作为待复核的开发表示。C0/C1均满足已标定分量ALL6非劣；12个final完整wave距离均值分别为0.15255321144257886 / 0.15042006897712382。C1的seed1-fast相对A改善1.9887788589535327%，因此全部6格至少2%的用途门未通过；权重未晋级，科学状态仍为response_match_not_established。

选择性回收142/318文件，176个原始NPZ仍在服务器。下一项是已登记的36对保存数组CPU复核，新增CP/NN/AD/训练/新随机库均0；当前仅已准备，真实启动以回执为准。

[开发比较](receipts/wp2_014_20261015/postrun_engineering_01/DEVELOPMENT_SUMMARY.md) · [原生终点与计数](receipts/wp2_014_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json) · [选择性回收](receipts/wp2_014_20261015/collection_01/COLLECTION_RECEIPT.json) · [读回配置](receipts/wp2_014_20261015/readback_registration_01/READBACK_EXECUTION.json)

以下保留此前阶段说明。

'''
readme.write_text(lead+readme.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
selector = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
d = read(selector); d['plan_sha256'] = sha(readme);write(selector,d)
write(OUT/'ROOT_REGISTRATION_RECEIPT.json', dict(schema='n01/wp2-readback-root-registration/v1',
    UTC=now, execution=ref, collection=cref, provisional_selection=sref,
    cumulative_training_updates=167667, active_scientific_jobs=0, remote_started=False,
    original95_source_unchanged=True, scientific_status='response_match_not_established'), True)
print(json.dumps(dict(status='readback_registered_not_started', configuration_sha256=ref['configuration_sha256'],
    source_controls=6, cumulative_training_updates=167667, active_scientific_jobs=0)))
