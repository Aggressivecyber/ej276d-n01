"""Verify collected small metadata and journal; never open original NPZ or host."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
COLLECT = STAGE / 'readback_collection_01'
TARGET = ROOT / 'runs/wp3_016_20261015/readback_collected_01'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def record(p):
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))

def write_new(path, value):
    with path.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

# Correct one local summary flag, retaining the exact earlier report and builder.
# This does not change any registered source, native evidence or historical time.
old_summary = HERE / 'START_VALIDATION_SUMMARY.json'
old_builder = HERE / 'build_start_validation.py'
start_summary = read(old_summary)
assert start_summary['passed'] is True
assert start_summary['resources']['GPU_usage_not_queried'] is False
before = HERE / 'before_GPU_query_flag_correction_01'
before.mkdir()
for p in (old_summary, old_builder):
    shutil.copyfile(p, before / p.name)
start_summary['resources']['GPU_usage_not_queried'] = True
start_summary['local_summary_flag_correction'] = dict(
    UTC=datetime.now(timezone.utc).isoformat(),
    field='resources.GPU_usage_not_queried', before=False, after=True,
    reason='The CPU controller did not query GPU usage; GPU work budget remains zero.',
    exact_previous_summary=record(before / old_summary.name),
    no_registered_or_native_source_changed=True)
old_summary.write_text(json.dumps(start_summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8', newline='\n')
source = old_builder.read_text(encoding='utf8')
assert source.count('GPU_usage_not_queried=False') == 1
old_builder.write_text(source.replace('GPU_usage_not_queried=False', 'GPU_usage_not_queried=True'), encoding='utf8', newline='\n')

reg = read(STAGE / 'readback_registration_01/READBACK_EXECUTION.json')
receipt = read(COLLECT / 'COLLECTION_RECEIPT.json')
remote = read(COLLECT / 'REMOTE_ARCHIVE.json')
archive = COLLECT / 'readback_output_01.tar'
assert receipt['status'] == 'sidecar_readback_outputs_verified'
assert receipt['partial_evidence_only'] is False
assert sha(archive) == remote['archive_sha256'] == receipt['archive_sha256']
assert archive.stat().st_size == remote['archive_bytes'] == 512000
inventory = remote['inventory']
assert len(inventory) == receipt['files'] == 19
assert set(p.relative_to(TARGET).as_posix() for p in TARGET.rglob('*') if p.is_file()) == set(inventory)
verified = {}
for name, expected in inventory.items():
    rel = PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts
    p = TARGET / name
    assert not p.is_symlink() and p.suffix not in ('.npz', '.pt', '.pth')
    actual = dict(bytes=p.stat().st_size, sha256=sha(p))
    assert actual == expected, name
    verified[name] = actual
with tarfile.open(archive) as bundle:
    members = bundle.getmembers()
    assert len(members) == 19 and {m.name for m in members} == set(inventory)
    for member in members:
        assert member.isfile() and member.size == inventory[member.name]['bytes']
        assert hashlib.sha256(bundle.extractfile(member).read()).hexdigest() == inventory[member.name]['sha256']
assert sum(v['bytes'] for v in inventory.values()) <= 80 * 1024**2

done = read(TARGET / 'artifacts/COMPLETE.json')
report = read(TARGET / 'artifacts/ARRAY_READBACK.json')
native = read(TARGET / 'supervisor/exit_receipt.json')
first = read(TARGET / 'artifacts/FIRST_PAIR_PROOF.json')
snapshot = read(STAGE / 'readback_start_verification_01/REMOTE_STATE.json')
expected = reg['expected_complete_counters']
assert len(expected) == 16
assert done['passed'] is report['passed'] is True
assert report['status'] == 'PASS' and done['native_status'] == 'COMPLETE'
assert native['exit_code'] == native['child_native_exit_code'] == 0
assert native['reason'] is native['error'] is None
assert not done['inflight'] and not report['inflight']
assert done['counters'] == done['attempted_counters'] == report['counters'] == report['attempted_counters'] == expected
assert done['operations_entered'] == done['operations_completed'] == report['operations_entered'] == report['operations_completed'] == reg['expected_operations'] == 200
assert sha(TARGET / 'artifacts/ARRAY_READBACK.json') == done['ARRAY_READBACK_sha256'] == '7337c5ab50929eb4381fc29ded4af138420b8257ba0c7b9945a7c82ca6be604a'
assert done == snapshot['artifacts/COMPLETE.json'] and native == snapshot['supervisor/exit_receipt.json']
assert first == snapshot['artifacts/FIRST_PAIR_PROOF.json']
assert first['counters']['charge_npz_files_decoded'] >= 2 and first['counters']['bank_pairs_verified'] >= 1

# Independent standard-library journal replay; no producer/worker imports.
raw_journal = (TARGET / 'artifacts/operations.jsonl').read_bytes()
assert raw_journal.endswith(b'\n')
events = [json.loads(line) for line in raw_journal.splitlines()]
assert len(events) == 400
entered, completed, inflight = {}, {}, {}
entered_counts = {k: 0 for k in expected}
completed_counts = {k: 0 for k in expected}
operation_types = Counter()
for event in events:
    index = event['operation_id']
    assert type(index) is int and 0 <= index < 200
    charges = event['charges']
    assert all(k in expected and type(v) is int and v >= 0 for k,v in charges.items())
    if event['event'] == 'entered':
        assert index == len(entered) and index not in entered
        entered[index] = event
        inflight[index] = event
        for k,v in charges.items(): entered_counts[k] += v
        operation_types[event['name']] += 1
    else:
        assert event['event'] == 'completed' and index in inflight and index not in completed
        original = inflight.pop(index)
        assert all(event[k] == original[k] for k in ('operation_id', 'name', 'charges', 'details'))
        completed[index] = event
        for k,v in charges.items(): completed_counts[k] += v
assert len(entered) == len(completed) == 200 and not inflight
assert entered_counts == completed_counts == expected
assert dict(operation_types) == dict(raw_NPZ_SHA=112, baseline100_decode=8,
    charge_bank_saved_statistics=8, hard_bank_saved_statistics=48, two_bank_pair_verified=24)
for key in ('entered_counts', 'completed_counts'):
    assert {k:native[key].get(k,0) for k in expected} == expected
assert native['unknown_inflight_ids'] == native['failed_ids'] == [] and native['truncated_tail'] is False
assert len(report['pairs']) == 24 and len(report['charge_banks']) == 8 and report['probes'] == []
assert all(v == 0 for v in reg['scientific_budget'].values())
assert all(v == 0 for v in report['reviewer_operations'].values())
limits = reg['resource_limits']
assert native['elapsed_seconds'] <= limits['wall_seconds']
assert native['peak_owned_RSS_bytes'] <= limits['RSS_MiB'] * 1024**2
assert report['resources']['peak_RSS_bytes'] <= limits['RSS_MiB'] * 1024**2
assert native['peak_worker_artifact_bytes'] <= limits['output_MiB'] * 1024**2

plan = read(TARGET / 'READBACK_PLAN.json')
raw_inventory = plan['raw_inventory']
assert len(raw_inventory) == 112
assert sum(v['bytes'] for v in raw_inventory.values()) == 2169655337
assert raw_inventory == report['raw_files_SHA_checked']
assert all(not (TARGET / name).exists() for name in raw_inventory)
result = dict(
    schema='n01/wp3-016-readback-collection-readback-summary/v1',
    recorded_UTC=datetime.now(timezone.utc).isoformat(), passed=True,
    status='NATIVE0_COMPLETE_SMALL_OUTPUTS_COLLECTED_ALL_SHA_AND_JOURNAL_VERIFIED',
    run_id=reg['run_id'], source_commit=start_summary['source_commit'],
    config=record(STAGE / 'readback_registration_01/READBACK_EXECUTION.json'),
    registered_sources=reg['files'], original_training_source_commit=reg['original_training_source_commit'],
    original_configuration_sha256=reg['original_configuration_sha256'],
    original_source_manifest_sha256=reg['original_source_manifest_sha256'],
    remote_root=reg['remote_root'], original_root=reg['original_root'],
    collected_root=str(TARGET), archive=record(archive),
    files_verified=len(verified), selected_content_bytes=sum(v['bytes'] for v in verified.values()),
    selected_inventory=verified, archive_all_regular_members_and_extracted_bytes_verified=True,
    collection_receipt=record(COLLECT / 'COLLECTION_RECEIPT.json'),
    remote_archive_receipt=record(COLLECT / 'REMOTE_ARCHIVE.json'),
    ARRAY_READBACK=record(TARGET / 'artifacts/ARRAY_READBACK.json'),
    terminal_complete=record(TARGET / 'artifacts/COMPLETE.json'),
    journal=record(TARGET / 'artifacts/operations.jsonl'),
    counters_completed=completed_counts, counters_entered=entered_counts,
    operations_entered=200, operations_completed=200, journal_events=400,
    operation_types=dict(operation_types), actual_inflight={}, unknown_inflight_ids=[], failed_ids=[], truncated_tail=False,
    native_exit=native, same_snapshot_health=start_summary['same_snapshot_health'],
    worker_complete_UTC=done['UTC'], supervisor_exit_UTC=native['UTC'],
    dispatch_UTC=start_summary['dispatch_UTC'], first_formal_pair_UTC=first['UTC'],
    only_explicit_snapshot_UTC=start_summary['only_snapshot_UTC'],
    supervisor_pid=native['supervisor_pid'], worker_pid=native['child_pid'],
    boot_id=start_summary['boot_id'], supervisor_start_ticks=start_summary['supervisor_start_ticks'], worker_start_ticks=start_summary['worker_start_ticks'],
    resources=dict(worker_getrusage=report['resources'],
        supervisor_sampled_peak_owned_RSS_bytes=native['peak_owned_RSS_bytes'],
        supervisor_elapsed_seconds=native['elapsed_seconds'],
        supervisor_peak_worker_artifact_bytes=native['peak_worker_artifact_bytes'],
        registered_limits=limits, current_process_RSS_VM_affinity_after_exit=None,
        GPU_work_budget=0, GPU_usage_not_queried=True,
        historical_last_resource_sample_does_not_override_terminal_journal=True),
    raw_remaining=dict(files=112, bytes=2169655337, inventory=raw_inventory,
        evidence='Worker each-file SHA once and decode once; raw files were read-only, omitted from collection and not rehashed locally or after audit.',
        raw_transferred=0, local_raw_decodes=0, local_raw_SHA_reads=0, raw_remote_deletion=False),
    scopes=dict(saved_ADC_hard_rows=1265856, baseline_files=8, charge_banks=8, hard_bank_pairs=24,
        actual_A384_rows=210976, actual_A_float64_cells=81014784, actual_mu_draws=210976, actual_q_draws=210976, parent_moment_records=52744,
        saved_input_digest_repetition_and_crossbank_identity_only=True,
        original_SOURCE_rehashed=False, teacher_or_charge_latent_RNG_replay=False,
        full_stochastic_preADC384_saved_or_replayed=False, full_CDF_loss_or_C1_median_replay=False, charge_r8_added=False),
    calls=dict(prepare_CLI=1, prepare_SSH=1, install_SSH=1, source_SCP=1,
        start_CLI_SSH=1, presence_wait_SSH=1, explicit_snapshot_CLI_SSH=1,
        collect_CLI_SSH=1, collection_SCP=1, total_SSH=6, total_SCP=2,
        collection_internal_terminal_validation_snapshot=1, additional_snapshot_CLI=0,
        status_progress_polls=0, automatic_retries=0, second_CPU_worker=0, extra_scientific_probe=0),
    scientific_budget=reg['scientific_budget'], training_counter_delta=0, cumulative_G_before_after=179667,
    prior_failed_native_costs_preserved_separately=dict(G=0, AD=0, A_rows=32965, head_noise_charge_qscalar_rows=26372,
        ADC_gather_rows=237348, CP_loads=3, charge_init_fit=1, optimizer_created=1),
    start_summary=record(old_summary), start_summary_flag_correction_only=True,
    no_more_host_calls=True, no_CPU_future_wake_required=True,
    scientific_acceptance=False, weights_promoted=False,
    next_action='Independent reviewer reads only collected small JSON/journal and applies the frozen research-reference rule; no repeat raw hashing/decoding.')
summary_path = HERE / 'COLLECTION_READBACK_SUMMARY.json'
write_new(summary_path, result)
md = f'''# 016 CPU读回回收闭合

一次CPU任务原生结束，child/supervisor均0；200项 entered=completed，400条journal逐项重聚合，16计数与冻结计划完全一致，inflight/failed为空。

19个小文件逐SHA核验，共{result['selected_content_bytes']:,}B；tar {archive.stat().st_size:,}B，SHA `{sha(archive)}`。ARRAY_READBACK为PASS，SHA `{sha(TARGET / 'artifacts/ARRAY_READBACK.json')}`。

112原始NPZ共2,169,655,337B只在登记CPU审计中各SHA/decode一次，原地保留，未传输或在本地重新解码。48HARD、48REPORT、8baseline、8charge；24双bankpair、1,265,856 hard行、210,976 actual A384/mu/q draw及52,744父moment记录完成。原SOURCE输入只核保存digest与lineage一致，不重读原SOURCE。

原生用时{native['elapsed_seconds']:.3f}s；worker getrusage峰RSS {report['resources']['peak_RSS_bytes']:,}B，supervisor采样峰RSS {native['peak_owned_RSS_bytes']:,}B。二者不同测量，均在1536MiB内；CPU2、VM3072MiB、输出与时限按登记约束，退出后当前RSS/VM/亲和性无值。

prepare/start/presence/explicit snapshot/collect各一次，SSH6/SCP2；collect按成熟控制器内置终点验证，未追加独立snapshot命令。所有远端调用停止，无CPU唤醒。CP/NN/AD/G/RNG/原TRAINDEV读取/GPU预算均0，累计G保持179667；原失败native非训练成本单列保留。科学接受及权重晋级仍为false，最终研究参考交独审按冻结规则确认。
'''
with (HERE / 'COLLECTION_READBACK_SUMMARY.md').open('x', encoding='utf8', newline='\n') as f:
    f.write(md)
print(json.dumps(dict(passed=True, summary=record(summary_path), start_summary=record(old_summary),
    files=19, selected_content_bytes=result['selected_content_bytes'], operations=200, journal_events=400,
    operation_types=dict(operation_types)), ensure_ascii=False))
