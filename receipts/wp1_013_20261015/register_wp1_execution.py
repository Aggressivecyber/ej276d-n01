"""Record the ready or verified-started WP1 state without scientific execution."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
RUN = 'wp1_013_20261015'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('ready', 'started'), required=True)
    parser.add_argument('--freeze-receipt', required=True)
    parser.add_argument('--start-receipt')
    parser.add_argument('--verification-receipt')
    parser.add_argument('--analysis-pid', type=int)
    parser.add_argument('--supervisor-pid', type=int)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).isoformat()
    cfg_path = ROOT / 'wp1_013_config.json'
    cfg = json.loads(cfg_path.read_text('utf-8'))
    assert cfg['status'] == 'frozen' and cfg['scientific_training_updates'] == 0
    assert cfg['budget_caps']['G_updates'] == cfg['budget_caps']['new_AD_queries'] == 0
    frozen = Path(args.freeze_receipt).resolve(strict=True)
    assert frozen.is_relative_to(ROOT)
    manifest_path = ROOT / 'WP1_SOURCE_MANIFEST.json'
    assert manifest_path.is_file()
    for old_name in ('MECHANISM012_SOURCE_MANIFEST.json', 'STABILITY011_SOURCE_MANIFEST.json', 'SHAPE010_SOURCE_MANIFEST.json'):
        old = json.loads((ROOT / old_name).read_text('utf-8'))
        assert all(digest(ROOT / rel) == sha for rel, sha in old['files'].items()), old_name
    states = [ROOT / 'MASTER_EXECUTION_PLAN.json', ROOT / 'PROGRAM.json', ROOT / 'jobs.json',
        Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
    snapshot = ROOT / ('receipts/' + RUN + '/metadata_before_' + args.stage)
    snapshot.mkdir(parents=True, exist_ok=False)
    for path in states + [ROOT / 'README.md']:
        shutil.copyfile(path, snapshot / path.name)
    values = [json.loads(path.read_text('utf-8')) for path in states]
    master, program, jobs, selector = values
    assert program['new_training_updates'] == 159667 and program['current_run_training_updates'] == 0
    job = next(row for row in jobs['jobs'] if row['job_id'] == 'N01_WP1_013_20261015')
    assert not any(row.get('scientific_job') and row.get('status') in ('running', 'native_running', 'started')
        for row in jobs['jobs'] if row is not job)
    live = args.stage == 'started'
    refs = {'configuration': 'wp1_013_config.json', 'configuration_sha256': digest(cfg_path),
        'source_manifest': 'WP1_SOURCE_MANIFEST.json', 'source_manifest_sha256': digest(manifest_path),
        'freeze_receipt': str(frozen.relative_to(ROOT)), 'freeze_receipt_sha256': digest(frozen),
        'observation_contract': 'OBSERVATION_CONTRACT.json', 'data_roles': 'DATA_ROLES.json',
        'acceptance_contract': 'docs/ACCEPTANCE_CONTRACT.md', 'budget_caps': cfg['budget_caps'],
        'resource_limits': cfg['resource_limits'], 'run_training_updates': 0, 'new_AD_queries': 0}
    if live:
        assert args.start_receipt and args.verification_receipt and args.analysis_pid and args.supervisor_pid
        for key, raw in [('start_receipt', args.start_receipt), ('startup_verification', args.verification_receipt)]:
            path = Path(raw).resolve(strict=True)
            assert path.is_relative_to(ROOT)
            refs[key] = str(path.relative_to(ROOT))
            refs[key + '_sha256'] = digest(path)
        refs.update(analysis_pid=args.analysis_pid, supervisor_pid=args.supervisor_pid, verified_correct_start=True)
    state = 'WP1_013_native_running' if live else 'WP1_013_frozen_ready_for_launch'
    master.update(status=state, active_work_package='WP1', active_scientific_jobs=int(live), updated_UTC=stamp,
        current_action='One scheduled wake to collect WP1 and advance the authorized master plan.' if live else 'Launch the one frozen WP1 run and verify its first family.')
    master['budgets']['WP1'].update(exact_NN_ADC_CP_budget_pending_implementation=False,
        budget_caps=cfg['budget_caps'], resources=cfg['resource_limits'])
    master['work_packages'][0].update(status='native_running' if live else 'frozen_ready', execution=refs)
    for value in (program, selector):
        value.update(status=state, active_scientific_jobs=int(live), active_research=RUN, current_work_package='WP1',
            current_stage='WP1_013_bounded_contract_validation', current_run_training_updates=0, updated_UTC=stamp,
            next_action=master['current_action'], scientific_status='response_match_not_established',
            wp1_execution=refs, master_plan_sha256=None)
    job.update(status='native_running' if live else 'frozen_ready_for_launch', exact_budget_pending=False,
        source_freeze_pending=False, execution=refs, budget_caps=cfg['budget_caps'], resources=cfg['resource_limits'],
        updated_UTC=stamp, analysis_pid=args.analysis_pid, supervisor_pid=args.supervisor_pid)
    if live:
        job['started_and_verified_UTC'] = stamp
    jobs.update(active_scientific_jobs=int(live), updated_UTC=stamp)
    save(states[0], master)
    for value in (program, selector):
        value['master_plan_sha256'] = digest(states[0])
    for path, value in zip(states[1:], (program, jobs, selector)):
        save(path, value)
    readme = (ROOT / 'README.md').read_text('utf-8')
    marker = '\n<!-- N01 WP1 CURRENT STATUS END -->\n'
    if marker in readme:
        historical = readme.split(marker, 1)[1]
    else:
        historical = readme.replace('# N01：012 新头梯度与裁窗机制诊断已完成', '## 历史完成节点：012 新头梯度与裁窗机制诊断', 1)
    header = '# N01：013 到项目交付主实施计划正在推进\n\n'
    header += '013 已在服务器启动并核验首次计算。' if live else '013 的观察、数据角色、实用误差合同和运行源码已经冻结，等待原生启动。'
    header += ' 本次训练更新和新 AD 查询均为 0；累计 N01 科学训练更新仍为 159667。\n\n'
    header += '主线保留确定性平均响应与输入相关随机响应，固定 256 点硬观察输出。六个工作包已统一登记：013 收口与评价合同、目标与代理优化、受控响应模块、已知响应恢复、稳健确认、软件与论文材料交付。短候选最多 12 个，教师试验最多 8 个，最终确认预算单列。\n\n'
    header += '当前 013 使用全部 6593 个 SOURCE TRAIN 父事件与 21047 条 REAL TRAIN 记录、两个独立随机库和三种电子学假设；仅验证 5 个新头参数方向及 3 个预定电压尺度，不因局部末位差异追加审计。原 DEV 保留已暴露历史；422 个原 REAL 组来自同一采集。\n\n'
    header += '工程可用、在域有效、跨条件物理复用分别判定。科学状态仍为 **response_match_not_established**；跨采集复用尚未验证。\n\n'
    header += '[用户主计划](USER_MASTER_PLAN_20261015.txt) · [工作包与预算](MASTER_EXECUTION_PLAN.json) · [观察合同](OBSERVATION_CONTRACT.json) · [数据角色](DATA_ROLES.json) · [实用误差合同](docs/ACCEPTANCE_CONTRACT.md) · [013 冻结配置](wp1_013_config.json) · [当前作业](jobs.json)\n\n'
    header += '以下节点保留形成时的结果与作业状态。\n' + marker
    (ROOT / 'README.md').write_bytes((header + historical).encode('utf-8'))
    receipt = {'schema': 'n01/wp1-execution-registration/v1', 'run_id': RUN, 'stage': args.stage, 'UTC': stamp,
        'status': state, 'active_scientific_jobs': int(live), 'execution': refs,
        'updated_metadata_sha256': {str(path): digest(path) for path in states + [ROOT / 'README.md']},
        'historical_training_updates': 159667, 'current_training_updates': 0,
        'protected_original_sources_unchanged': True, 'scientific_status': 'response_match_not_established'}
    out = ROOT / ('receipts/' + RUN + '/EXECUTION_' + args.stage.upper() + '.json')
    with out.open('xb') as handle:
        handle.write((json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
