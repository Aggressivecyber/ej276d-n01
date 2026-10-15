"""Local stdlib-only015 controller identities and Audit journal bridge; no arrays."""
from __future__ import annotations
import argparse
import base64
import contextlib
import copy
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


def require(test, message):
    if not test:
        raise AssertionError(message)


def rejects(function, message):
    try:
        function()
    except (ValueError, AssertionError):
        return
    raise AssertionError(message)


def module(path, name):
    compile(path.read_bytes(), str(path), 'exec')
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    require(not out.exists() and out.drive.upper() in ('E:', 'D:'), 'fresh E/D fixture')
    out.mkdir(parents=True)
    (out/'tmp').mkdir()
    os.environ.update(TEMP=str(out/'tmp'), TMP=str(out/'tmp'), PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    base = repo/'receipts/wp3_016_20261015'
    source = base/'readback_engineering_01/readback_control.py'
    c = module(source, 'readback016_control_artificial')
    directory = base/'postrun_independent_01'
    sources = {'readback_control.py': source, 'readback_saved_arrays.py': directory/'readback_saved_arrays.py',
        'saved_array_kernels.py': directory/'saved_array_kernels.py', 'READBACK_PLAN.json': directory/'READBACK_PLAN.json',
        'BOUND_ORIGINAL_INVENTORY.json': base/'collection_01/REMOTE_FULL_INVENTORY.json'}
    plan = c.read(sources['READBACK_PLAN.json'])
    ready = c.read(directory/'READBACK_WORKER_READY.json')
    for name, label in [('readback_saved_arrays.py', 'worker'), ('saved_array_kernels.py', 'kernel'), ('READBACK_PLAN.json', 'plan')]:
        require(c.record(sources[name]) == {k: ready[label][k] for k in ('bytes', 'sha256')}, 'ready mismatch: '+name)
    require(c.PLAN_SHA == c.sha(sources['READBACK_PLAN.json']), 'current plan pin mismatch')
    require(set(c.FILES) == set(sources) and len(sources)+1 == 6, 'six source/control slots')
    require(all(v == 0 for v in plan['scientific_budget'].values()), 'nonzero scientific budget')
    for name in ('readback_saved_arrays.py', 'saved_array_kernels.py'):
        compile(sources[name].read_bytes(), str(sources[name]), 'exec')
    frozen = c.read(repo/'WP3_016_SOURCE_MANIFEST.json')
    require(len(frozen['files']) == 116, 'frozen116 source count')
    for name, row in frozen['files'].items():
        require(c.record(repo/name) == row, 'frozen source changed: '+name)
    require(c.sha(repo/'wp3_016_config.json') == c.CONFIG016_SHA
        and c.sha(repo/'WP3_016_SOURCE_MANIFEST.json') == c.MANIFEST016_SHA, 'frozen controls changed')
    passed = ['current_worker_kernel_plan_ready_and_frozen116_plus2_controls_SHA_verified']

    p = dict(schema='n01/wp3-independent-readback-execution/v1', run_id=c.RUN_ID, remote_root=c.SIDE,
        original_root=c.ORIGINAL, resource_limits=c.LIMITS, automatic_retry=False, original_files_read_only=True,
        files={n: dict(path=c.SIDE+'/'+n, local_path=str(path), **c.record(path)) for n, path in sources.items()})
    c.validate(p)
    bad = copy.deepcopy(p)
    bad['resource_limits'] = dict(c.LIMITS, CPU_threads=3)
    rejects(lambda: c.validate(bad), 'changed CPU cap accepted')
    bad = copy.deepcopy(p)
    bad['remote_root'] = c.ORIGINAL+'/forbidden_sidecar'
    rejects(lambda: c.validate(bad), 'output inside original accepted')
    passed.append('exact_five_sources_plus_registration_and_finite_resource_contract')

    # Exercise actual gate against E-drive copies; each of six changed slots fails.
    gated = out/'identity_gate_side'
    gated.mkdir()
    for name, path in sources.items():
        shutil.copyfile(path, gated/name)
    gp = copy.deepcopy(p)
    original_side = c.SIDE
    c.SIDE = str(gated)
    gp['remote_root'] = c.SIDE
    for name in gp['files']:
        gp['files'][name]['path'] = c.SIDE+'/'+name
    c.new_json(gated/c.CONFIG, gp)
    config_sha = c.sha(gated/c.CONFIG)
    try:
        c.gate(config_sha)
        for name in [*sources, c.CONFIG]:
            path = gated/name
            original_bytes = path.read_bytes()
            path.write_bytes(original_bytes+b'\n')
            try:
                rejects(lambda: c.gate(config_sha), 'changed source/control slot accepted: '+name)
            finally:
                path.write_bytes(original_bytes)
        c.gate(config_sha)
    finally:
        c.SIDE = original_side
    passed.append('actual_gate_all_six_source_control_identity_tamper_rejections')

    # Same exact report metadata paths consumed by Audit.one_bank; local selected JSON only.
    inventory = c.read(sources['BOUND_ORIGINAL_INVENTORY.json'])
    reports = sorted(n[:-4]+'.json' for n in inventory['omitted_remote_inventory'] if n.endswith('_REPORT.npz'))
    require(len(reports) == 48, 'report metadata count')
    selected = repo/'runs/wp3_016_20261015/selected_collected_01'
    for name in reports:
        require(c.record(selected/name) == inventory['whole_remote_inventory'][name], 'selected report identity: '+name)
    passed.append('all_48_consumed_REPORT_JSON_bound_to_original211_inventory')

    prior_argv, prior_native, seen = sys.argv, c.native, []
    try:
        payload = dict(mode='prepare', config_sha256='a'*64, controller_sha256=c.sha(source), token='0'*32)
        sys.argv = [str(source), '--bootstrap', base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()]
        c.native = lambda value: (seen.append(value) or {'status': 'synthetic_bootstrap_parsed'})
        with contextlib.redirect_stdout(io.StringIO()):
            require(c.main() == 0, 'bootstrap parser failed')
        require(seen == [payload], 'bootstrap payload changed')
    finally:
        sys.argv, c.native = prior_argv, prior_native
    passed.append('actual_bootstrap_parser_no_remote_action')

    # Reuse actual Audit writer only. All charged events below are synthetic annotations.
    w = module(sources['readback_saved_arrays.py'], 'readback016_writer_artificial')
    original = out/'manufactured_empty_original'
    original.mkdir()
    side = out/'manufactured_side'
    side.mkdir()
    audit = w.Audit(SimpleNamespace(run_root=original, out=side/'artifacts', artificial_only=True))
    audit.inventory_sha = c.sha(sources['BOUND_ORIGINAL_INVENTORY.json'])
    for row in inventory['omitted_remote_inventory'].values():
        audit.op('SYNTHETIC_SHA_COUNT_NO_IO', dict(raw_files_SHA_checked=1, raw_bytes_SHA_read=row['bytes']), lambda: None)
    for _ in range(2):
        audit.op('SYNTHETIC_BASELINE_NO_ARRAY', dict(baseline_npz_files_decoded=1), lambda: None)

    def fake_charge():
        audit.op('SYNTHETIC_CHARGE_NO_ARRAY',dict(charge_npz_files_decoded=1,charge_A_rows_verified=26372,
            charge_A_float64_cells_verified=26372*384,charge_mu_draws_verified=26372,charge_q_draws_verified=26372,
            charge_parent_moment_records_verified=6593),lambda:None)
    for _ in range(2):fake_charge()

    def fake_pair():
        for _ in range(2):
            audit.op('SYNTHETIC_BANK_NO_ARRAY', dict(hard_rows_files_decoded=1, report_npz_files_decoded=1,
                waveform_rows_decoded=26372), lambda: None)

    audit.op('SYNTHETIC_PAIR_NO_ARRAY', dict(bank_pairs_verified=1), fake_pair)
    proof = dict(audit.base(), schema='n01/wp3-readback-first-formal-pair/v1', actual_saved_array_pair_completed=True,
        no_extra_tiny_actual_input=True, actual_new_CP_NN_AD=0)
    w.write_new(audit.out/'FIRST_PAIR_PROOF.json', proof)
    svpid, wpid, regsha = 900000, os.getpid(), 'a'*64
    worker_command = [c.PYTHON, '-B', c.SIDE+'/readback_saved_arrays.py', '--run-root', c.ORIGINAL,
        '--inventory', c.SIDE+'/BOUND_ORIGINAL_INVENTORY.json', '--out', c.SIDE+'/artifacts']
    state = {'START.json': dict(boot_id='synthetic', supervisor_pid=svpid, supervisor_start_ticks=111, registration_sha256=regsha),
        'supervisor/launch.json': dict(supervisor_pid=svpid, child_pid=wpid, supervisor_start_ticks=111,
            child_start_ticks=222, boot_id='synthetic', command=worker_command),
        'current_boot_id': 'synthetic', 'registration_sha256': regsha, 'supervisor/exit_receipt.json': None,
        'artifacts/FIRST_PAIR_PROOF.json': copy.deepcopy(proof), 'artifacts/status.json': audit.base(), 'artifacts/FAILED.json': None,
        'durable_progress': c.legacy().journal_tools().durable_progress(audit.out),
        'supervisor/resource_status.json': dict(resource_limits=c.LIMITS, owned_process_tree=[svpid, wpid],
            elapsed_seconds=10, RSS_bytes=2048, worker_artifact_bytes=4096),
        'live_processes': {
            'supervisor': dict(alive=True, pid=svpid, start_ticks=111, cwd=c.SIDE, VmRSS=1024,
                command=[c.PYTHON, '-B', c.SIDE+'/readback_control.py', '--supervise', '--config-sha256', regsha, '--lock-fd', '3']),
            'worker': dict(alive=True, pid=wpid, parent_pid=svpid, start_ticks=222, cwd=c.SIDE, VmRSS=1024,
                VmSize=2048, command=worker_command, CPU_affinity=[0, 1])}}
    rejects(lambda: c.first_valid(state, p), 'artificial first pair blessed')
    state['artifacts/FIRST_PAIR_PROOF.json']['artificial_only'] = False  # Only this in-memory copy.
    c.first_valid(state, p)
    for kind in ('failed_file', 'status_failed', 'native_nonzero', 'unknown_proof', 'dead_worker', 'reused_pid',
                 'wrong_parent', 'wrong_command', 'RSS_excess', 'VM_excess', 'bad_journal', 'truncated_journal', 'status_behind'):
        bad = copy.deepcopy(state)
        if kind == 'failed_file': bad['artifacts/FAILED.json'] = {'error': 'manufactured'}
        if kind == 'status_failed': bad['artifacts/status.json']['native_status'] = 'FAILED'
        if kind == 'native_nonzero': bad['supervisor/exit_receipt.json'] = {'child_native_exit_code': 1, 'exit_code': 1}
        if kind == 'unknown_proof': bad['artifacts/FIRST_PAIR_PROOF.json']['inflight'] = {'x': {}}
        if kind == 'dead_worker': bad['live_processes']['worker']['alive'] = False
        if kind == 'reused_pid': bad['live_processes']['worker']['start_ticks'] = 999
        if kind == 'wrong_parent': bad['live_processes']['worker']['parent_pid'] = 1
        if kind == 'wrong_command': bad['live_processes']['worker']['command'] = ['unrelated']
        if kind == 'RSS_excess': bad['supervisor/resource_status.json']['RSS_bytes'] = 1537*1024**2
        if kind == 'VM_excess': bad['live_processes']['worker']['VmSize'] = 3073*1024**2
        if kind == 'bad_journal': bad['durable_progress']['failed_ids'] = [999]
        if kind == 'truncated_journal': bad['durable_progress']['truncated_tail'] = True
        if kind == 'status_behind': bad['artifacts/status.json']['counters']['bank_pairs_verified'] = 0
        rejects(lambda bad=bad: c.first_valid(bad, p), 'invalid current health accepted: '+kind)
    busy = copy.deepcopy(state)
    busy['durable_progress']['unknown_inflight_ids'] = [999]
    require(c.first_valid(busy, p)['current_inflight_not_counted_complete'] == [999], 'normal inflight incorrectly rejected/counted')
    passed.append('actual_Audit_writer_first_pair_identity_and_13_same_snapshot_failure_guards')

    for _ in range(6):
        audit.op('SYNTHETIC_BASELINE_NO_ARRAY', dict(baseline_npz_files_decoded=1), lambda: None)
    for _ in range(23):
        audit.op('SYNTHETIC_PAIR_NO_ARRAY', dict(bank_pairs_verified=1), fake_pair)
    for _ in range(6):fake_charge()
    require(audit.entered == audit.completed == 200, 'finite016 operation count')
    w.write_new(audit.out/'ARRAY_READBACK.json', dict(artificial_only=True, synthetic_counter_annotations_not_IO=True, actual_arrays=0))
    complete = dict(audit.base(), native_status='COMPLETE', passed=True,
        ARRAY_READBACK_sha256=c.sha(audit.out/'ARRAY_READBACK.json'), scientific_acceptance=False)
    w.write_new(audit.out/'COMPLETE.json', complete)
    shutil.copyfile(sources['BOUND_ORIGINAL_INVENTORY.json'], side/'BOUND_ORIGINAL_INVENTORY.json')
    state.update({'artifacts/COMPLETE.json': copy.deepcopy(complete), 'artifacts/status.json': audit.base(),
        'supervisor/exit_receipt.json': dict(child_native_exit_code=0, exit_code=0, reason=None, error=None,
            supervisor_pid=svpid, child_pid=wpid, elapsed_seconds=10, peak_owned_RSS_bytes=2048, peak_worker_artifact_bytes=4096),
        'durable_progress': c.legacy().journal_tools().durable_progress(audit.out)})
    old_side = c.SIDE
    c.SIDE = str(side)
    try:
        rejects(lambda: c.complete_valid(state, p), 'artificial complete blessed')
        state['artifacts/COMPLETE.json']['artificial_only'] = False  # Only this in-memory copy.
        c.complete_valid(state, p)
        bad = copy.deepcopy(state)
        bad['artifacts/COMPLETE.json']['counters']['bank_pairs_verified'] = 23
        rejects(lambda: c.complete_valid(bad, p), 'incomplete finite scope blessed')
        bad = copy.deepcopy(state)
        bad['durable_progress']['unknown_inflight_ids'] = [999]
        rejects(lambda: c.complete_valid(bad, p), 'terminal unknown operation blessed')
        bad = copy.deepcopy(state)
        bad['artifacts/COMPLETE.json'] = None
        rejects(lambda: c.first_valid(bad, p), 'native0 without completed producer accepted')
    finally:
        c.SIDE = old_side
    passed.append('actual_Audit_200_operations_terminal_count_journal_and_native_completion_bridge')
    require(not {'torch', 'numpy', 'n01'} & set(sys.modules), 'numerical package imported')
    receipt = dict(schema='n01/wp3-readback-controller-engineering/v1', UTC=datetime.now(timezone.utc).isoformat(),
        artificial_only=True, passed=True, passed_count=len(passed), passed_checks=passed,
        controller=c.record(source), worker=c.record(sources['readback_saved_arrays.py']), kernel=c.record(sources['saved_array_kernels.py']),
        plan=c.record(sources['READBACK_PLAN.json']), inventory=c.record(sources['BOUND_ORIGINAL_INVENTORY.json']),
        checker=c.record(Path(__file__)), producer_writer_imported=True, numerical_packages_imported=False,
        synthetic_counter_annotations_not_IO=True, positive_native_markers_in_memory_only=True,
        actual_operations=dict(original_CP_reads=0, raw_NPZ_reads=0, array_decodes=0, NN_rows=0, AD_queries=0,
            new_random_draws=0, training_updates=0, SSH_calls=0), original116_plus2_controls_unchanged=True)
    c.new_json(out/'ENGINEERING_RECEIPT.json', receipt)
    print(json.dumps(dict(status='readback016_stdlib_interfaces_passed', checks=len(passed),
        controller_sha256=c.sha(source), receipt_sha256=c.sha(out/'ENGINEERING_RECEIPT.json'))))


if __name__ == '__main__':
    main()
