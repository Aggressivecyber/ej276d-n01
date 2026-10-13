"""In-memory native/prelaunch receipt fixtures; no native run or remote call."""
from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from metadata_audit import ROOT, OUT, sha

sys.path.insert(0, str(ROOT / 'scripts'))
import window008_resume01_runtime as runtime


def main():
    source = ROOT / 'scripts/window008_resume01_runtime.py'
    digest = sha(source)
    cfg = runtime.read(ROOT / 'window008_resume01_config.json')
    base = runtime.read(ROOT / cfg['base_configuration'])
    parent = runtime.read(ROOT / cfg['fix01_source_manifest'])
    rec = ROOT / 'receipts/window008_resume01_20261013'
    qa_path = (rec / 'engineering_cuda_01/ENGINEERING_RECEIPT.json').resolve()
    native_path = (rec / 'engineering_supervisor_01/exit_receipt.json').resolve()
    launch_path = (rec / 'engineering_supervisor_01/launch.json').resolve()
    prelaunch_path = (rec / 'PRELAUNCH_REVIEW.json').resolve()
    core = {name: sha(ROOT / name) for name in ['src/n01/response008_resume01.py', 'src/n01/train_window008_resume01.py',
                                               'scripts/check_window008_resume01_cuda.py', 'scripts/window008_resume01_runtime.py']}
    fixtures = {
        qa_path: dict(status='passed_native_resume01_engineering_and_import', device='cuda:0', pid=203, parent_pid=202,
                      started_UTC='2026-10-13T15:40:03+00:00', finished_UTC='2026-10-13T15:40:04+00:00',
                      scientific_G_updates=0, scientific_D_updates=0, actual_SOURCE_forward_rows=0,
                      actual_REAL_forward_rows=0, actual_SOURCE_numeric_decodes=0, actual_REAL_numeric_decodes=0,
                      DEV_numeric_decodes=0, engineering_G_updates=3, engineering_D_updates=6,
                      artificial_forward_rows=6144, artificial_forward_calls=96,
                      config_sha256=sha(ROOT / 'window008_resume01_config.json'), source_sha256=core,
                      periodic_import=dict(status='passed', source_checkpoint_sha256=cfg['reference_inputs']['fix01_full_periodic_06000']['sha256'],
                                           source_checkpoint_purpose='periodic', G_updates=6000,
                                           D_updates=dict(fast=3000, slow=3000, common_level=3000),
                                           new_G_updates=0, new_D_updates=dict(fast=0, slow=0, common_level=0),
                                           native_tree_bitwise_equal=True, optimizer_moments_reset=False),
                      steps=[dict(arm='full', stage='joint_stochastic', peak_GPU_bytes=1024),
                             dict(arm='simple', stage='mean_response', peak_GPU_bytes=1024),
                             dict(arm='simple', stage='joint_stochastic', peak_GPU_bytes=1024)]),
        native_path: dict(exit_code=0, reason=None, peak_rss_bytes=1024, supervisor_pid=201, child_pid=202,
                          started_utc='2026-10-13T15:40:02+00:00', ended_utc='2026-10-13T15:40:05+00:00'),
        launch_path: dict(supervisor_pid=201, child_pid=202, started_utc='2026-10-13T15:40:02+00:00',
                          command=['timeout', '--signal=TERM', '--kill-after=15s', '300s', '/usr/bin/python', '-B',
                                   'scripts/check_window008_resume01_cuda.py', '--out', cfg['remote_root'] + '/runs/engineering_cuda_01']),
        prelaunch_path: dict(status='passed_independent_prelaunch_review', files_sha256=core),
    }
    originals = dict(read=runtime.read, validate_inputs=runtime.validate_inputs)
    results = []

    def activate(data):
        def read(path):
            key = Path(path).resolve()
            return copy.deepcopy(data[key]) if key in data else originals['read'](path)
        runtime.read = read
        runtime.validate_inputs = lambda root=ROOT: (cfg, base, parent)

    activate(fixtures)
    try:
        runtime.validate_evidence(ROOT)
    except (ValueError, KeyError, TypeError) as error:
        results.append(dict(case='positive_native3G6D6144_rows_fixture', passed=False, error=str(error)))
    else:
        results.append(dict(case='positive_native3G6D6144_rows_fixture', passed=True))

    def reject(name, alter):
        candidate = copy.deepcopy(fixtures)
        alter(candidate)
        activate(candidate)
        try:
            runtime.validate_evidence(ROOT)
        except (ValueError, KeyError, TypeError) as error:
            results.append(dict(case=name, passed=True, rejected=True, rejection=str(error)))
        else:
            results.append(dict(case=name, passed=False, rejected=False))

    reject('native_exit1_rejected', lambda d: d[native_path].update(exit_code=1))
    reject('native_reason_failure_rejected', lambda d: d[native_path].update(reason='rss_limit'))
    reject('native_RSS_cap_rejected', lambda d: d[native_path].update(peak_rss_bytes=9*1024**3))
    reject('native_parent_PID_chain_rejected', lambda d: d[native_path].update(child_pid=999))
    reject('native_QA_parent_chain_rejected', lambda d: d[qa_path].update(parent_pid=999))
    reject('native_QA_pid_equals_wrapper_rejected', lambda d: d[qa_path].update(pid=202))
    reject('native_QA_time_before_launch_rejected', lambda d: d[qa_path].update(started_UTC='2026-10-13T15:39:00+00:00'))
    reject('native_QA_device_CPU_rejected', lambda d: d[qa_path].update(device='cpu'))
    reject('native_scientific_G_updates_rejected', lambda d: d[qa_path].update(scientific_G_updates=1))
    reject('native_actual_SOURCE_forward_rejected', lambda d: d[qa_path].update(actual_SOURCE_forward_rows=1))
    reject('native_actual_REAL_decode_rejected', lambda d: d[qa_path].update(actual_REAL_numeric_decodes=1))
    reject('native_actual_SOURCE_decode_rejected', lambda d: d[qa_path].update(actual_SOURCE_numeric_decodes=1))
    reject('native_DEV_decode_rejected', lambda d: d[qa_path].update(DEV_numeric_decodes=1))
    reject('native_artificial_G4_rejected', lambda d: d[qa_path].update(engineering_G_updates=4))
    reject('native_artificial_D_missing_rejected', lambda d: d[qa_path].update(engineering_D_updates=3))
    reject('native_artificial_rows_wrong_rejected', lambda d: d[qa_path].update(artificial_forward_rows=8192))
    reject('native_artificial_steps_missing_rejected', lambda d: d[qa_path]['steps'].pop())
    reject('native_GPU_cap_rejected', lambda d: d[qa_path]['steps'][0].update(peak_GPU_bytes=11*1024**3))
    reject('native_import_not_passed_rejected', lambda d: d[qa_path]['periodic_import'].update(status='failed'))
    reject('native_wrong_config_SHA_rejected', lambda d: d[qa_path].update(config_sha256='0'*64))
    reject('native_wrong_source_SHA_rejected', lambda d: d[qa_path]['source_sha256'].update({'src/n01/response008_resume01.py': '0'*64}))
    reject('native_wrong_output_root_rejected', lambda d: d[launch_path]['command'].__setitem__(-1, '/unbound/out'))
    reject('prelaunch_not_passed_rejected', lambda d: d[prelaunch_path].update(status='pending'))
    reject('prelaunch_missing_core_rejected', lambda d: d[prelaunch_path]['files_sha256'].pop('src/n01/train_window008_resume01.py'))
    reject('prelaunch_changed_core_rejected', lambda d: d[prelaunch_path]['files_sha256'].update({'src/n01/response008_resume01.py': '0'*64}))
    runtime.read = originals['read']
    runtime.validate_inputs = originals['validate_inputs']
    assert sha(source) == digest, 'Runtime helper changed during native receipt fixture review'
    report = dict(schema='n01/window008-resume01-independent-native-receipt-fixtures/v1',
                  UTC=datetime.now(timezone.utc).isoformat(), source_sha256=digest,
                  status='passed' if all(row['passed'] for row in results) else 'failed', cases=results,
                  artificial_JSON_receipts_only=True, actual_native_run_verified=False,
                  scientific_model_forwards=0, optimizer_updates=0, GPU_calls=0, SSH_calls=0,
                  DEV_numeric_decodes=0, project_state_writes=0)
    suffix = sha(source)[:12]
    (OUT / ('NATIVE_HELPER_RECEIPT_FIXTURES_' + suffix + '.json')).write_text(json.dumps(report, indent=2) + '\n', 'utf8')
    print(json.dumps(dict(status=report['status'], cases=len(results), source_sha256=digest,
                          failed=[row for row in results if not row['passed']]), indent=2))


if __name__ == '__main__':
    main()
