from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib,json

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
timeout=json.loads((HERE/'PREFLIGHT_TIMEOUT.json').read_bytes())
deployment=json.loads((HERE/'PRE_DISPATCH_REVIEW.json').read_bytes())
phases=[json.loads(v) for v in (HERE/'NATIVE_PREFLIGHT_STDOUT.jsonl').read_bytes().splitlines()]
assert timeout['seconds']==240 and not timeout['scientific_dispatch_attempted']
assert len(phases)==1 and phases[0]['stage']=='EXISTING_SCIENTIFIC_PATH_PREFLIGHT_STARTED'
assert not (HERE/'NATIVE_PREFLIGHT.json').exists() and not (HERE/'DISPATCH_RECEIPT.json').exists()
assert deployment['source_files_verified']==193 and deployment['small_files_written']==6
assert not Path('C:/Users/10629/.codex/automations/n01/automation.toml').exists()
now=datetime.now(timezone.utc).isoformat();local=datetime.fromisoformat(now).astimezone(timezone(timedelta(hours=8)))
rel=HERE.relative_to(ROOT).as_posix()
record=dict(status='REGISTERED_FIXED_CANDIDATE_NOT_DISPATCHED_NATIVE_PREFLIGHT_TIMEOUT',UTC=now,
    native_existing_path_preflight_timeout_seconds=240,last_native_phase=phases[-1],
    native_source_verification_completed=False,native_import_verification_completed=False,
    exact_blocking_file_or_system_cause='unknown; no storage diagnosis or repair performed',
    Windows_existing_path_sources_verified=193,small_files_written=6,small_bytes_written=deployment['small_bytes_written'],
    scientific_dispatch_attempted=False,scientific_generator_calls=0,actual_candidate_evaluations=0,
    parameter_updates=0,active_scientific_jobs=0,followup_count=0,
    followup_basis='No running science job or measured completion time; previous single wake deleted and absence verified.',
    storage_owner='Other agent assigned by user; this chat does not change NAS mounts/connections/transport/migration.',
    next_action='After an external-state change makes the existing scientific paths available, retain this failed preflight attempt, run one new bounded preflight into fresh receipt files, then dispatch the already registered fixed check unchanged.',
    scientific_acceptance=False)
(HERE/'EXECUTION_BLOCKER.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
for filename in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    path=ROOT/filename;d=json.loads(path.read_bytes())
    assert d['active_scientific_jobs']==0 and d['pending_registered_job_id']=='JOINT_RESPONSE_01_RECENTERED_CANDIDATE_CHECK_20261020'
    d.update(status='JOINT_RESPONSE_01_fixed_candidate_not_dispatched_native_preflight_timeout',updated_UTC=now,
        execution_state_asof_UTC=now,active_scientific_jobs=0,active_job_id=None,
        current_action='742 unique derivatives and both fixed restoration KKT certificates completed. Fixed-candidate code/audit and193 Windows source bindings ready; native existing-path preflight timed out240s before source-complete/import proof. No science dispatch.',
        next_action=record['next_action'],active_scientific_jobs_status_basis='Prior scientific process exited0; local solves exited0; next scientific dispatch never attempted.',
        scientific_acceptance=False)
    d['joint_response_01'].update(recentered_candidate_check_status='not_dispatched_native_preflight_timeout',
        recentered_candidate_check_blocker=rel+'/EXECUTION_BLOCKER.json',
        recentered_candidate_check_local_preparation=rel+'/PRE_DISPATCH_REVIEW.json',
        recentered_candidate_check_source_files_verified_Windows=193,recentered_candidate_check_source_files_verified_native=False,
        recentered_candidate_check_scientific_dispatch_attempted=False,recentered_candidate_check_execution_counts={},
        current_scientific_acceptance=False)
    if filename=='jobs.json':
        assert len(d['jobs'])==145
        job=d['jobs'][-1];assert job['job_id']=='JOINT_RESPONSE_01_RECENTERED_CANDIDATE_CHECK_20261020'
        job.update(status='not_dispatched_native_preflight_timeout',verified_UTC=now,
            execution_blocker=rel+'/EXECUTION_BLOCKER.json',local_preparation=rel+'/PRE_DISPATCH_REVIEW.json',
            source_files_verified_Windows=193,source_files_verified_native=False,dispatched=False,
            generator_calls=0,actual_candidate_evaluations=0,parameter_updates=0,scientific_acceptance=False)
    path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
header=f'''# N01 当前阶段：导数与两头恢复KKT审核通过，固定候选待启动

> 最后核验（{local:%Y-%m-%d %H:%M}，UTC+8）：742个唯一导数列及两头各一次恢复求解/独立KKT审核通过。固定候选实际检查已登记和准备，计算端现有路径核验240秒超时，尚未派发。0运行科学作业、0定时。阶段4科学验收未过，阶段5未进入。

[导数审核](receipts/joint_response_01_20261020/recentered_derivative_completion_01/terminal_01/TERMINAL_REVIEW.json)复算308数组、71680条角色身份、742列、84风险视图及14控制门；192源/25输出、计数和资源界通过。job142退出0，新增424列与旧318列合并。原job141失败及9列/20生成重复成本保留。

[恢复独立审核](receipts/joint_response_01_20261020/recentered_restoration_solve_fix01/INDEPENDENT_AUDIT.json)重建两组10709×53矩阵，两头修正范数0.00207434534/0.00144677842，半径0.003125；驻点约1.60e-15/5.65e-15、互补约6.98e-11/5.39e-11。两次求解耗时47.64秒、峰RSS135512064字节，未执行生成或提交更新。job143的错误绝对零检查在求解前失败；最小fix01精确保留约1e-20噪声坐标，噪声改变量零，旧失败/源码/配置保留。

[固定候选协议](docs/JOINT_RESPONSE_01_RECENTERED_CANDIDATE_CHECK_20261020.md)只允许两头各一候选、14库、42生成调用，旧中心/控制重放及原joint/native/trust门不改，无追加求解或参数提交。执行和独立审核代码齐备，现有Windows路径上193源/192pins核验通过，6份小文本已写入研究目录。全部七库已参与约束拟合，诊断不替代独立物理验收。

[启动阻碍回执](receipts/joint_response_01_20261020/recentered_candidate_check_01/EXECUTION_BLOCKER.json)：计算端预检仅返回开始阶段，240秒内未返回源文件完成/原生导入通过证明；具体阻塞文件或系统原因未知，未派发科学作业。NAS挂载、连接和传输管理由用户指定的另一agent负责，本聊天不做修复。等待外部状态改变后沿用已登记候选和原预算继续；无运行计算或可估算完成时刻，因此不保留轮询定时。以下历史快照保留，当前以本节、三台账与回执为准。

'''
path=ROOT/'README.md';old=path.read_text(encoding='utf-8')
path.write_text(header+'---\n'+old.split('---\n',1)[1],encoding='utf-8',newline='\n')
path=ROOT/'JOINT_RESPONSE_01_PLAN.md';lines=path.read_text(encoding='utf-8').splitlines()
lines[2]=header.splitlines()[2]
path.write_text('\n'.join(lines)+'\n\n## 固定候选登记与原生启动核验阻碍\n\n'+header.split('\n\n',2)[2],encoding='utf-8',newline='\n')
print(json.dumps(record))
