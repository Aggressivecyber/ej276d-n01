from datetime import datetime,timezone
from pathlib import Path
import json,shutil

ROOT=Path('E:/GAN/research/N01')
HERE=Path(__file__).resolve().parent
before=HERE/'before_public_state_01';before.mkdir()
for name in ('README.md','JOINT_RESPONSE_01_PLAN.md','PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    shutil.copy2(ROOT/name,before/name)
prefix='receipts/joint_response_01_20261020/random_precision_diagnostic_01/'
now=datetime.now(timezone.utc).isoformat()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    path=ROOT/name;d=json.loads(path.read_bytes())
    assert d['active_scientific_jobs']==1 and d['active_job_id']=='JOINT_RESPONSE_01_RANDOM_PRECISION_DIAGNOSTIC_20261020'
    assert d['followup']['count']==1 and d['followup']['next_wake_local']=='2026-10-20T12:29:00+08:00'
    d['updated_UTC']=now
    d['current_action']='Fixed-endpoint paired IID32/antithetic-Z32 precision diagnostic correctly started; first completed bank and native-runtime random replay audited.'
    d['next_action']='At the single 2026-10-20 12:29 Asia/Singapore wake, consume the schedule, collect terminal evidence and audit all 864 FD columns and 36 fixed checks, including exact native-runtime RNG replay. If still active within the original budget, estimate one new wake from measured remaining work.'
    d['joint_response_01']['random_precision_diagnostic_prefix_review']=prefix+'AUDITOR_PREFIX_REVIEW.json'
    d['joint_response_01']['random_precision_diagnostic_RNG_platform_review']=prefix+'AUDITOR_RNG_PLATFORM_DIFFERENCE.json'
    d['joint_response_01']['random_precision_diagnostic_complete_terminal_audit_executed']=False
    path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
path=ROOT/'README.md';old=path.read_text(encoding='utf-8');head,tail=old.split('\n---\n',1)
assert head.startswith('# N01 当前阶段：阶段4固定曲率序列闭合，科学门未过')
new='''# N01 当前阶段：阶段4随机精度诊断已启动，12:29单次接续

> 当前执行状态（2026-10-20）：三轮REAL校准及五项固定方向诊断已完成并审核，原最终科学门尚未通过。新随机精度诊断已正确启动；当前登记1科学作业、1次苏醒。REAL响应匹配尚未建立，阶段5待验。

[原三轮校准](receipts/joint_response_01_20261020/original_precision_REAL_pilot_01/terminal_01/TERMINAL_REVIEW.json)完成600差分列、20联合解及192最终门复算，两神经各24/96改善门失败。五项固定方向检查均零参数提交；[最后一次曲率检查](receipts/joint_response_01_20261020/second_curvature_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)仍有seed0 FRESH、seed1 FIT0初值QPSD的1.010563/1.010417超过1.01。同一中心的三候选曲率序列已经停止。

[保存几何审计](receipts/joint_response_01_20261020/second_curvature_diagnostic_01/SAVED_ANCHOR_AND_GEOMETRY_DIAGNOSIS.json)发现初值锚余量不足和弱方向敏感性。已按[冻结协议](docs/JOINT_RESPONSE_01_RANDOM_PRECISION_DIAGNOSTIC_20261020.md)比较四组独立随机库、两头的IID32与正负成对潜变量32，组内共享电子学噪声；共864差分列、1800生成调用，只评价固定端点和既有候选，零新求解或更新。

[启动与估时](receipts/joint_response_01_20261020/random_precision_diagnostic_01/WAIT_ESTIMATE.json)于11:35:45（UTC+8）核验879源文件和原进程身份；保存进度为63/864列、136/1800生成调用。按本次实测1.664秒/调用估计12:21:36完成，加15%剩余时间余量并向上取整，已设置**2026-10-20 12:29（UTC+8）单次接续**，见[定时回执](receipts/joint_response_01_20261020/random_precision_diagnostic_01/AUTOMATION_CREATE_RECEIPT.json)。后续进度尚未查询，原5400秒内部上限保持。

[首组审核](receipts/joint_response_01_20261020/random_precision_diagnostic_01/AUDITOR_PREFIX_REVIEW.json)复算54列差分及两个固定预测。Windows Torch 2.10与Linux Torch 2.12的正态数值最大相差2个float32 ULP；[原环境独立重放](receipts/joint_response_01_20261020/random_precision_diagnostic_01/AUDITOR_RNG_PLATFORM_DIFFERENCE.json)的262144值逐位一致，均匀随机数及生成器状态哈希相同。完整终止审计将以原环境精确重放核验全部8组随机库，科学输入和门限保持原样。

以下保留历史快照；当前状态以本节、三台账及对应回执为准。

'''
path.write_text(new+'\n---\n'+tail,encoding='utf-8',newline='\n')
path=ROOT/'JOINT_RESPONSE_01_PLAN.md';s=path.read_text(encoding='utf-8')
old='> 当前执行状态（2026-10-20）：三轮REAL校准及五项固定方向诊断已闭合负结果；新四组随机精度诊断已前瞻登记，当前0科学作业、0定时。原科学门未通过，阶段5尚未进入。'
assert s.count(old)==1
s=s.replace(old,'> 当前执行状态（2026-10-20）：三轮REAL校准及五项固定方向诊断已闭合负结果；新四组随机精度诊断已正确启动，当前登记1科学作业、1次苏醒，2026-10-20 12:29（UTC+8）接续。原科学门未通过，阶段5尚未进入。')
s+='''
## 2026-10-20 随机精度诊断启动、首组审核及单次接续

[正确启动](receipts/joint_response_01_20261020/random_precision_diagnostic_01/STARTUP_ASSESSMENT.json)确认879源/82pins、原进程身份、旧中心/锚与候选复现。11:35:45（UTC+8）[估时快照](receipts/joint_response_01_20261020/random_precision_diagnostic_01/WAIT_ESTIMATE.json)核验136生成调用、63/864差分列；启动到进度226.325秒，剩余1664调用及10秒落盘按实测预计12:21:36完成，15%剩余时长余量后向上取整至12:29。单次[定时配置回读](receipts/joint_response_01_20261020/random_precision_diagnostic_01/AUTOMATION_CREATE_RECEIPT.json)通过，仅本聊天一项活动苏醒；没有持续查询。

[首组保存算术审核](receipts/joint_response_01_20261020/random_precision_diagnostic_01/AUDITOR_PREFIX_REVIEW.json)通过54列J/N与两固定候选预测。首次本地随机数逐位断言失败记录保留：[差异核验](receipts/joint_response_01_20261020/random_precision_diagnostic_01/AUDITOR_RNG_PLATFORM_DIFFERENCE.json)表明Windows Torch 2.10与原Linux Torch 2.12的262144个正态值中2591个相差1—2 float32 ULP；原环境独立重放全部逐位一致，两端均匀随机数和生成器状态哈希相同。审计改为原环境精确重放全部8组已保存库并核验文件/数组哈希，本地使用原库复算矩阵；不扩大数值容差，不改科学代码、输入、接受门或运行预算。完整终止审核尚未执行。
'''
path.write_text(s,encoding='utf-8',newline='\n')
print(json.dumps(dict(status='PUBLIC_STATE_UPDATED',UTC=now,scientific_acceptance=False,execution_state_asof_UTC='2026-10-20T03:35:45.698756+00:00',next_wake_local='2026-10-20T12:29:00+08:00')))
