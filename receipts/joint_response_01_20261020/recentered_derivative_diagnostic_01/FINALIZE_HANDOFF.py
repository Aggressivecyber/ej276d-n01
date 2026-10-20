from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import tomllib

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
rel = HERE.relative_to(ROOT).as_posix()
now = datetime.now(timezone.utc).isoformat()
raw = (HERE/'AUTOMATION_CONFIGURATION.toml').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'ec083510c947448fcc82b5386baf1df3a85abe361f446a727acca6ad09af1670'
automation = tomllib.loads(raw.decode('utf-8'))
estimate = json.loads((HERE/'WAIT_ESTIMATE.json').read_bytes())
receipt = dict(status='SINGLE_WAKE_CREATED_AND_READBACK_VERIFIED', UTC=now,
    automation_id='n01', single_wake_local=estimate['single_wake_local'],
    single_wake_UTC=estimate['single_wake_UTC'], active_automations_for_this_thread=['n01'],
    target_thread_id=automation['target_thread_id'], kind='heartbeat', count=1,
    configuration_sha256=hashlib.sha256(raw).hexdigest(),
    prompt_exact_readback=True, scientific_acceptance=False)
receipt_path = HERE/'AUTOMATION_CREATE_RECEIPT.json'
receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((ROOT/name).read_bytes())
    assert d['active_scientific_jobs'] == 1 and d['active_job_id'] == estimate['job_id']
    assert not d['scientific_acceptance']
    d.update(updated_UTC=now,
        next_action='At the measured single wake, audit NAS terminal evidence, then perform only the registered once-per-head fresh-J restoration feasibility solve if eligible.',
        followup_automation_id='n01', followup_automation_status='ACTIVE',
        followup_next_run_local=estimate['single_wake_local'],
        followup_scheduled_local=estimate['single_wake_local'],
        next_wake_local=estimate['single_wake_local'],
        followup_receipt=rel+'/AUTOMATION_CREATE_RECEIPT.json',
        followup_receipt_sha256=receipt_hash,
        current_followup_status='SINGLE_WAKE_WAITING_ON_BOUNDED_RECORDED_JOB',
        followup_status='ACTIVE_SINGLE_WAKE_MEASURED')
    d['followup'] = dict(status='ACTIVE_SINGLE_WAKE', automation_id='n01', count=1,
        next_wake_local=estimate['single_wake_local'], receipt=rel+'/AUTOMATION_CREATE_RECEIPT.json',
        wait_estimate=rel+'/WAIT_ESTIMATE.json')
    (ROOT/name).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

readme = ROOT/'README.md'
old = readme.read_text(encoding='utf-8')
assert old.startswith('# N01 当前阶段：阶段4固定弱方向投影消融闭合')
header = """# N01 当前阶段：阶段4新中心导数诊断已启动

> 当前执行状态（2026-10-20 13:22，UTC+8）：1个科学作业，已核验71/742差分列；14:02单次接续。REAL响应匹配尚未建立，阶段5未进入。

[新诊断协议](docs/JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_DIAGNOSTIC_20261020.md)固定两头原控制候选为新中心，在3个FIT和4个已用IID库重新测53个响应坐标的完整J/N。噪声固定，原模型、loss、硬读出与门限不变；本作业不求新候选、不提交参数。所有新运行输出和缓存直接写入NAS `/mnt/MiNas`，Windows对应 `Z:/`。现有数据迁移由另一任务负责。

[启动核验](receipts/joint_response_01_20261020/recentered_derivative_diagnostic_01/STARTUP_ASSESSMENT.json)确认185部署文件、进程身份及两组新旧中心重放。13:21:51进度为146/1512生成调用、71/742差分列；该快照之后的运行状态未观察。[实测估时](receipts/joint_response_01_20261020/recentered_derivative_diagnostic_01/WAIT_ESTIMATE.json)为13:56:25完成，加约335秒余量安排[14:02单次接续](receipts/joint_response_01_20261020/recentered_derivative_diagnostic_01/AUTOMATION_CREATE_RECEIPT.json)。内部3600秒、外部3900秒加20秒、4CPU、GPU4GiB、RSS12GiB预算不延长。

[已闭合固定投影诊断](receipts/joint_response_01_20261020/stable_projection_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)67.34秒正常结束，保存数值审核通过但两头局部门失败。投影使两头接受组数从原控制各8/9降至4/9、5/9；joint作用弱的方向仍承载原生保护，不按joint稳定性单独删减，不搜索阈值。原控制各一组QPSD初值比1.010505/1.010555超过原1.01门。

[新中心可行性筛查](receipts/joint_response_01_20261020/stable_projection_diagnostic_01/RECENTERED_RESTORATION_SCREEN.json)以已测控制输出为截距、旧中心导数为近似，在固定0.003125半径和1.005原生设计余量下两头均局部可行。旧导数过时，不能据此执行候选；必须先审核本次新导数，再按已登记设置各头至多一次求解及独立复算约束/KKT，通过后另行登记实际检查。

随机精度、同中心曲率和固定投影的历史失败保持。原三轮REAL两神经各24/96最终改善门失败，原5%native改善等科学门不变。代码、配置和汇总证据可公开；波形、逐行身份、NPZ和权重不上传。以下保留历史快照，当前以本节、三台账和对应回执为准。

"""
readme.write_text(header+'---\n'+old.split('---\n', 1)[1], encoding='utf-8')
plan = ROOT/'JOINT_RESPONSE_01_PLAN.md'
text = plan.read_text(encoding='utf-8')
before = '> 当前执行状态（2026-10-20）：随机精度和固定弱方向投影诊断均已审核失败；当前分析原控制候选新中心的可行性恢复。0科学作业、0定时；原科学门未过，阶段5未进入。'
after = '> 当前执行状态（2026-10-20 13:22，UTC+8）：新中心完整导数诊断已启动，71/742差分列核验；1科学作业、14:02单次接续。新输出直接写MiNas，原科学门未过，阶段5未进入。'
assert before in text
text = text.replace(before, after, 1)
text += """

## 2026-10-20 新中心导数诊断

随机精度和固定弱方向投影诊断均已审核闭合为负结果。未投影控制两头各8/9库通过，剩余失败是各一库QPSD初值比约1.0105超过原1.01门；固定投影使原生保护进一步恶化。保存证据表明旧中心native导数对移除方向的作用预测失真。

以已测控制作为新中心的旧导数筛查仅支持进一步测量。现已登记并启动[新中心导数诊断](docs/JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_DIAGNOSTIC_20261020.md)，固定两头、七个已用库、53响应坐标及±0.004差分；零噪声修正、零新候选和参数提交。精确预算742差分、1512生成调用49545216行、4536观察148635648行，输出直接保留NAS。13:22核验185部署文件、两组中心重放、71差分列和146生成调用。

按实测剩余量预计13:56:25结束，14:02单次接续；预算与余量见[估时回执](receipts/joint_response_01_20261020/recentered_derivative_diagnostic_01/WAIT_ESTIMATE.json)。终止审核通过后才在保存新J/N上按固定0.003125半径、1.005原生设计余量各头至多一次最小范数修正求解，复算全部约束与KKT。两头均通过才另行登记真实候选检查；不搜索阈值/半径，不将局部可行替代原科学验收。
"""
plan.write_text(text, encoding='utf-8')
print(json.dumps(dict(status='SINGLE_WAKE_AND_CURRENT_DOCUMENTS_RECORDED', automation_receipt_sha256=receipt_hash,
    wake_local=estimate['single_wake_local'], scientific_acceptance=False)))
