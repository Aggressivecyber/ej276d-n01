from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
rel = HERE.relative_to(ROOT).as_posix()
startup = json.loads((HERE/'STARTUP_ASSESSMENT.json').read_bytes())
wait = json.loads((HERE/'WAIT_ESTIMATE.json').read_bytes())
auto = json.loads((HERE/'AUTOMATION_READBACK.json').read_bytes())
dispatch = json.loads((HERE/'DISPATCH_RECEIPT.json').read_bytes())
assert startup['status'] == 'PASS_CORRECT_NATIVE_COMPLETION_STARTUP'
assert auto['status'] == 'PASS_SINGLE_WAKE_CONFIGURATION_READBACK' and auto['current_chat_active_count'] == 1
assert wait['scheduled_wake_local'] == auto['scheduled_wake_local'] == '2026-10-20T16:07:00+08:00'
now = datetime.now(timezone.utc).isoformat()
receipt = dict(status='RUNNING_STARTUP_VERIFIED_AND_SINGLE_WAKE_RECORDED', UTC=now,
    scientific_snapshot_UTC=startup['progress_snapshot_UTC'], source_verification_UTC=startup['source_verification_snapshot_UTC'],
    active_jobs_at_last_verified_snapshot=1, job_id='JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_COMPLETION_20261020',
    new_FD_columns_last_verified=27, new_generator_calls_last_verified=56,
    old_saved_columns_reused=318, scientific_budget_extended=False,
    automation_id='n01', followup_count=1, wake_local=wait['scheduled_wake_local'],
    startup_sha256=hashlib.sha256((HERE/'STARTUP_ASSESSMENT.json').read_bytes()).hexdigest(),
    wait_sha256=hashlib.sha256((HERE/'WAIT_ESTIMATE.json').read_bytes()).hexdigest(),
    automation_readback_sha256=hashlib.sha256((HERE/'AUTOMATION_READBACK.json').read_bytes()).hexdigest(),
    config_sha256=dispatch['config_sha256'], terminal_audit_executed=False, scientific_acceptance=False,
    later_scientific_state_unknown=True, further_monitoring_performed=False)
(HERE/'FOLLOWUP_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    path = ROOT/name
    d = json.loads(path.read_bytes())
    assert d['active_job_id'] == receipt['job_id'] and d['active_scientific_jobs'] == 1
    assert d['followup']['count'] == 0
    d.update(status='JOINT_RESPONSE_01_recentered_derivative_completion_running', updated_UTC=now,
        execution_state_asof_UTC=startup['progress_snapshot_UTC'], active_scientific_jobs=1,
        current_stage='4_recentered_derivative_completion', active_work_package='4_recentered_derivative_completion',
        current_action='Correct native completion startup verified at07:40:37UTC:27/424 new derivatives,56/864 generator calls;318 saved columns reused. Later progress not polled.',
        next_action='At16:07UTC+8 collect terminal status through the private NAS session, audit combined742 unique derivatives, then follow the registered restoration gate.',
        active_scientific_jobs_status_basis='One process group verified with boot/PID/start ticks and actual derivative progress at the stated snapshot; later state is unknown.',
        current_followup_status='SCHEDULED_SINGLE_WAKE_VERIFIED', followup_status='SCHEDULED_SINGLE_WAKE_VERIFIED',
        followup_automation_id='n01', followup_automation_status='ACTIVE',
        followup_next_run_local=wait['scheduled_wake_local'], next_wake_local=wait['scheduled_wake_local'],
        followup_scheduled_local=wait['scheduled_wake_local'],
        followup_receipt_sha256=hashlib.sha256((HERE/'FOLLOWUP_RECEIPT.json').read_bytes()).hexdigest(), scientific_acceptance=False)
    d['followup'] = dict(status='SCHEDULED_SINGLE_WAKE_VERIFIED', automation_id='n01', count=1,
        next_wake_local=wait['scheduled_wake_local'], receipt=rel+'/FOLLOWUP_RECEIPT.json',
        estimate=rel+'/WAIT_ESTIMATE.json', configuration_readback=rel+'/AUTOMATION_READBACK.json')
    d['joint_response_01'].update(recentered_derivative_completion_status='running_startup_verified',
        recentered_derivative_completion_verified_UTC=startup['progress_snapshot_UTC'],
        recentered_derivative_completion_startup=rel+'/STARTUP_ASSESSMENT.json',
        recentered_derivative_completion_wait=rel+'/WAIT_ESTIMATE.json',
        recentered_derivative_completion_execution_counts_last_verified=startup['progress']['execution_counts'],
        recentered_derivative_completion_new_FD_columns_last_verified=27,
        recentered_derivative_completion_saved_prefix_columns_reused=318,
        recentered_derivative_completion_complete_terminal_audit_executed=False,
        recentered_derivative_combined_complete=False, current_scientific_acceptance=False)
    if name == 'jobs.json':
        job = d['jobs'][-1]
        assert len(d['jobs']) == 142 and job['job_id'] == receipt['job_id']
        job.update(status='running_startup_verified', verified_UTC=startup['progress_snapshot_UTC'],
            science_pid=231736, science_start_ticks='5524922', supervisor_pid=231687,
            startup=rel+'/STARTUP_ASSESSMENT.json', execution_counts_last_verified=startup['progress']['execution_counts'],
            wait_estimate=rel+'/WAIT_ESTIMATE.json', followup=rel+'/FOLLOWUP_RECEIPT.json',
            complete_terminal_audit_executed=False, scientific_acceptance=False)
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
local = datetime.fromisoformat(startup['progress_snapshot_UTC']).astimezone(timezone(timedelta(hours=8)))
path = ROOT/'README.md'
old = path.read_text(encoding='utf-8')
header = f'''# N01 当前阶段：固定导数补全已正确启动

> 最后核验状态（{local:%Y-%m-%d %H:%M}，UTC+8）：新增27/424差分、56/864生成调用，旧318列只读复用；1个科学作业。单次苏醒为今天16:07。之后进度未轮询。阶段4科学验收未过，阶段5未进入。

原job141因NAS写入中断，保存6库318列已独立审核；9列执行未落盘、415列未执行。原失败、原预算及未知退出码保留。[固定补全](docs/JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_COMPLETION_20261020.md)只新增8库424列、864生成；两次累计751实际列、742唯一列、1532生成，重复9列和20生成明确记账。

[启动审核](receipts/joint_response_01_20261020/recentered_derivative_completion_01/STARTUP_ASSESSMENT.json)已核验192源/191pins、12前缀、boot/PID/启动时刻、实际生成与导数进度及资源。新增1库、噪声32768行、REAL拟合4096行，旧/新中心重放、初值锚复用和控制各1；峰RSS3004895232字节。无求解、候选、参数提交、CHECK/DEV或分类器。

[独立NAS连接](receipts/joint_response_01_20261020/recentered_derivative_completion_01/PRIVATE_NAS_SESSION_POLICY.json)在N01自己的挂载命名空间使用同一NAS目录，保留原输入哈希、模型、loss、硬读出和科学预算；全局挂载与另一任务的迁移未改。共享访问及工程校验失败回执保留。新科学输出、缓存和临时数组直接写NAS。

[单次苏醒估算](receipts/joint_response_01_20261020/recentered_derivative_completion_01/WAIT_ESTIMATE.json)：原生距离单独计时84.090秒/56调用，余808调用、397列；该计时不含生成、特征和NAS I/O。以原外层16:05:49 TERM/16:06:09 KILL设定加回执余量向上取整至16:07核收，原1800/2100秒预算不延长。完整终止与合并742列审计尚未执行；其通过后才进入已登记的每头一次恢复可行性求解，局部解不当作实际候选或科学成功。

私有波形、逐行身份、NPZ和权重不上传。以下保留历史快照，当前以本节、三台账及回执为准。

'''
path.write_text(header+'---\n'+old.split('---\n', 1)[1], encoding='utf-8', newline='\n')
path = ROOT/'JOINT_RESPONSE_01_PLAN.md'
lines = path.read_text(encoding='utf-8').splitlines()
lines[2] = f'> 最后核验状态（{local:%Y-%m-%d %H:%M}，UTC+8）：固定补全已正确启动，新增27/424列、56/864生成，旧318列复用；1科学作业，今天16:07单次苏醒。之后状态未知。阶段4科学验收未过，阶段5未进入。'
text = '\n'.join(lines)+'\n'
text = text.replace('## 当前外部前提：NAS访问恢复', '## 14:54 NAS访问中断历史记录（已接续）', 1)
text += '''

## 15:40 固定补全原生启动与单次接续

同一已登记固定补全于15:30派发，15:40核验新增27/424差分、56/864生成、1库与6库318列复用，峰RSS3004895232字节；boot/PID/start ticks、192源/191pins和12前缀均通过。通过只属于N01进程的独立NAS连接完成部署、导入和执行，NAS目录、输入哈希与科学预算不改；全局挂载及既有迁移未改。此前工程超时保留，文件绑定与原位导入分别约70/80秒通过。

原生距离单项计时约1.502秒/调用，余808调用、397列；生成、特征及NAS I/O未单独计时。按原外层TERM16:05:49/KILL16:06:09设定加回执余量，安排16:07单次苏醒；不增加原1800/2100秒预算。此后未轮询。终止审计经同一独立连接在NAS原位执行，哈希I/O并行4，数值判据不改；240秒审计/300秒SSH仅是工程审核界，不是科学计算延长。见[启动审核](receipts/joint_response_01_20261020/recentered_derivative_completion_01/STARTUP_ASSESSMENT.json)、[等待估算](receipts/joint_response_01_20261020/recentered_derivative_completion_01/WAIT_ESTIMATE.json)和[单次定时回读](receipts/joint_response_01_20261020/recentered_derivative_completion_01/AUTOMATION_READBACK.json)。合并742唯一列尚未审核，恢复求解仍未触发，阶段4科学验收未过。
'''
path.write_text(text, encoding='utf-8', newline='\n')
print(json.dumps(receipt))
