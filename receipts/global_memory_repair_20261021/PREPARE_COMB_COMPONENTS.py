from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import shlex
import subprocess
import tarfile

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
out = here / 'comb_components_01'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
template_path = root / 'configs/global_memory_conditional_wave_20261021.json'
template = json.loads(template_path.read_bytes())
remote = '/home/rylan/projects/ej276d-n01/comb_components_20261021'
config = {k: template[k] for k in ('frozen_setup_config', 'roles_file', 'candidate_arrays')}
config.update(schema='n01/frozen-comb-component-diagnostic/v1',
    job_id='N01_COMB_COMPONENT_DIAGNOSTIC_20261021', registered_UTC=stamp,
    authorization='2026-10-21 user: 你先去调查那个波浪是怎来的',
    frozen_candidate=template['replay_checkpoint'], parent_chunk=128, bank_seed=2026102157,
    stages=['SOURCE', 'filter_only', 'time_only', 'residual_no_time', 'A0',
        'A0_fixed_residual_condition', 'A0_fixed_time_condition', 'current_mean',
        'current_z0', 'current_random', 'current_plus_noise'],
    modes=['clean_dynamic', 'clean_fixed126', 'ADC_dynamic'],
    attribution_contract=dict(
        evidence='Job148 localized oscillation before electronics and ADC to frozenA0; its internal mean components were not ablated.',
        formula='A0 = qbar*(W+R)/canonical_Q(W+R), W=warped filtered unit-peak SOURCE times onset gate, R=.20*tanh(mean_head(decoded))*gate.',
        component_controls='Remove neural residual; remove learned time warp; hold residual conditioning at logQ=logAmplitude=0; hold time conditioning at the same reference. Keep all source shapes, roles and qbar.',
        fixed_window='Only a parallel diagnostic:126 peak =>110:366 integration and156:366 tail. Original dynamic readout remains displayed.',
        scalar_scaling='Positive scalar qbar/canonical_Q does not change clean PSD; report exact fixed-window W/R decomposition and charge signs.',
        source_groups='Reuse previously declared source PSD centers0.207919414 and0.312575162 with tolerance1e-6; all other164 rows retained in full-population plots. Check full normalized shape dispersion within these groups.',
        trend_summaries='128 equal-count source-Q ordered slices per dominant group, median PSD and its total variation; descriptive resolution declared before results, not tuned to REPORT.',
        stochastic_scope='One fixed random draw per parent, plus matched fast electronic noise; used for chain visualization only, not uncertainty estimates.',
        success='Exact A0 reconstruction; immutable weights; identify whether fixed-window oscillation survives the component controls. No response acceptance or training claim.'),
    operation_budget=dict(checkpoint_loads=6, TRAIN_archives_decoded=1, PCA_archives_decoded=1,
        old_A_calls=52, old_A_rows=6593, M0_head_calls=104, M0_head_rows=13186,
        repair_checkpoint_loads=1, fixed_condition_UNet_calls=51, fixed_condition_UNet_rows=6465,
        current_mean_rows=6465, current_response_calls=102, current_response_rows=12930,
        noise005_calls=51, noise005_rows=6465, extension_rows=19395,
        observation_rows=213345),
    resources=dict(CPU_threads=4, GPU_GiB=4, RSS_GiB=12, internal_seconds=300,
        hard_wall_seconds=420, GPU_free_required_MiB=6144, output_MiB=64),
    storage=dict(remote_root=remote, policy='Active diagnostic arrays on server; temporary E analysis copy removed after plots. No large duplicate backup.'),
    optimizer_updates=0, new_trained_candidates=0, report_rows_used=0,
    NAS_file_IO_checks=False, NAS_file_hash_verification=False,
    stop_rule='One frozen full-population pass; no optimizer, finite-difference scan, tuning or automatic restart.',
    measured_budget_basis=dict(prior_job148_seconds=12.4481, expected_seconds=60, conservative_seconds=120,
        note='Extra fixed-condition U-Net and current response/noise stages; current shared-GPU time must be measured.'))
path = root / 'configs/comb_components_20261021.json'
assert not path.exists()
path.write_bytes((json.dumps(config, ensure_ascii=False, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
ast.parse((root / 'scripts/diagnose_joint01_comb_components.py').read_text())
snapshot = out / 'before_registration'
snapshot.mkdir()
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    raw = (root / name).read_bytes()
    (snapshot / name).write_bytes(raw)
    d = json.loads(raw)
    assert d['active_scientific_jobs'] == 0
    d.update(status='N01_COMB_component_diagnostic_registered', active_job_id=None, active_run_id=None,
        pending_job_id=config['job_id'], pending_registered_job_id=config['job_id'],
        updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='User redirected to root-cause investigation; frozen component attribution registered with0 updates.',
        next_action='Locate comb contribution from residual conditioning, time warp and readout; no training.')
    d['global_memory_repair'].update(status='comb_component_diagnostic_registered',
        current_committed_training_updates=0, config=str(path.relative_to(root)), config_sha256=sha,
        automatic_training_continuation=False, next_step_design=str(path.relative_to(root)))
    if name == 'jobs.json':
        assert len(d['jobs']) == 162
        d['jobs'].append(dict(job_id=config['job_id'], status='registered', registered_UTC=stamp,
            config=str(path.relative_to(root)), config_sha256=sha, kind='frozen_comb_component_causal_diagnostic',
            operation_budget=config['operation_budget'], resources=config['resources'], optimizer_updates=0,
            scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf8', newline='\n') as stream:
    stream.write('\n用户要求先调查PSD波浪来源。登记job163固定权重拆分：保留全部6465源父，'
        '分开滤波、均值时间变换、神经残差和两条能量条件路径，正式动态读出与固定126窗并列；'
        '再对照当前均值、零潜变量、一次随机响应、同一次电子学噪声和ADC。'
        '仅归因，0优化更新；不把固定窗或条件冻结诊断当作可交付模型。\n')
old_sha = hashlib.sha256(template_path.read_bytes()).hexdigest()
for name in ('REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py'):
    code = (here / 'conditional_wave_01' / name).read_text()
    code = code.replace(template['storage']['remote_root'], remote).replace(old_sha, sha)
    code = code.replace(template_path.name, path.name).replace('run_global_memory01.py', 'diagnose_joint01_comb_components.py')
    code = code.replace("'660s'", "'420s'")
    (out / name).write_bytes(code.encode())
collector = (here / 'COLLECT_CONDITIONAL_WAVE.py').read_text()
collector = collector.replace("'conditional_wave_01'", "'comb_components_01'").replace(template['storage']['remote_root'], remote)
(here / 'COLLECT_COMB_COMPONENTS.py').write_bytes(collector.encode())
with tarfile.open(out / 'runtime.tar', 'w') as archive:
    for member in list((root / 'src/n01').glob('*.py')) + list((root / 'scripts').glob('*.py')) + [path]:
        archive.add(member, arcname=member.relative_to(root).as_posix())
deploy = """import sys,tarfile
from pathlib import Path
root=Path(REMOTE)
root.mkdir(parents=True,exist_ok=False)
(root/'cache').mkdir()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive: archive.extractall(root,filter='data')
print('RUNTIME_DEPLOYED')
""".replace('REMOTE', repr(remote))
with (out / 'runtime.tar').open('rb') as stream:
    subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', 'rylan@100.82.231.102',
        '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(deploy)],
        stdin=stream, capture_output=True, check=True, timeout=60)
receipt = dict(status='REGISTERED_AND_DEPLOYED', UTC=stamp, job_number=163, config_sha256=sha,
    optimizer_updates=0, source_parents=6465, report_rows_used=0)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps(receipt))
