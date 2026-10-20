from datetime import datetime,timezone
from pathlib import Path
import copy,json,shutil

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
review=json.loads((HERE/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
terminal=json.loads((HERE/'terminal_01/REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status']=='PASS_SAVED_RANDOM_PRECISION_DIAGNOSTIC_AUDIT' and not review['two_head_precision_gate_pass']
assert not terminal['processes'] and terminal['native_exit']['exit_code']==0
assert json.loads((HERE/'WAKE1229_AUTOMATION_CONSUMED.json').read_bytes())['active_for_this_thread']==[]
before=HERE/'before_close_wake1229';before.mkdir();now=datetime.now(timezone.utc).isoformat()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(ROOT/name,before/name)
rel=HERE.relative_to(ROOT).as_posix()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    path=ROOT/name;d=json.loads(path.read_bytes())
    if name=='jobs.json':
        old=copy.deepcopy(d['jobs']);assert len(old)==139 and old[-1]['job_id']=='JOINT_RESPONSE_01_RANDOM_PRECISION_DIAGNOSTIC_20261020'
        d['jobs'][-1].update(status='completed_audited_precision_gates_failed',native_exit_code=0,ended_UTC=review['native_exit']['ended_utc'],execution_counts=review['execution_counts'],terminal_review=rel+'/terminal_01/TERMINAL_REVIEW.json',two_head_precision_gate_pass=False,parameter_updates=0)
        assert d['jobs'][:-1]==old[:-1]
    d.update(status='JOINT_RESPONSE_01_random_precision_diagnostic_closed_negative',active_scientific_jobs=0,active_job_id=None,active_run_id=None,pending_job_id=None,training_pid=None,supervisor_pid=None,current_job_native_exit_code=0,current_job_ended_UTC=review['native_exit']['ended_utc'],last_verified_UTC=terminal['UTC'],execution_state_asof_UTC=terminal['UTC'],updated_UTC=now,current_action='Paired latent antithetic diagnostic audited negative at the registered two-head precision gates.',next_action='Use saved repeated matrices and fixed candidates to separate random derivative variability from systematic local prediction error before another prospective diagnostic.',followup=dict(status='CONSUMED_DELETED',automation_id='n01',count=0,next_wake_local=None,receipt=rel+'/WAKE1229_AUTOMATION_CONSUMED.json'))
    d['joint_response_01'].update(random_precision_diagnostic_status='completed_audited_precision_gates_failed',random_precision_diagnostic_completed=True,random_precision_diagnostic_complete_terminal_audit_executed=True,random_precision_diagnostic_terminal_review=rel+'/terminal_01/TERMINAL_REVIEW.json',random_precision_diagnostic_execution_counts=review['execution_counts'],random_precision_diagnostic_two_head_gate_pass=False)
    path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
paragraph='''
## 2026-10-20 12:29接续：随机精度诊断完成，成对抽样门失败

[终止独立审核](receipts/joint_response_01_20261020/random_precision_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)确认879源/82pins/53输出、41,275,960字节、exit0、1764.60秒；864差分列、36固定接受判定、162风险视图、32预测误差、131072正负配对及8组噪声哈希全部复算。原Linux运行时精确重放8库共2097152个潜变量值，原科学代码/输入/门限未改，零新候选、零更新。

正负成对/IID的J离散度比为0.67150/0.87855，native wave/QPSD导数离散度比为0.86858/0.80620。两头native均未达到≤0.75，第二头J也失败。平均J范数比1.00111/0.99070；平均绝对native预测误差由0.002963/0.003385降至0.002669/0.003112，但这项局部改善不满足完整前瞻门，不启用成对训练。原REAL最终科学门仍未通过。

实际于12:00:56（UTC+8）结束，较按启动负载估计的12:21更早；12:29定时已删除并确认本聊天无活动定时，当前0科学作业。继续用保存的重复矩阵区分随机波动和系统局部模型误差，尚未登记新生成任务。
'''
path=ROOT/'JOINT_RESPONSE_01_PLAN.md';s=path.read_text(encoding='utf-8');lines=s.splitlines()
lines[2]='> 当前执行状态（2026-10-20）：随机精度诊断已完成并审核，两头成对抽样精度门失败；正在分析保存的随机波动与局部模型偏差。当前0科学作业、0定时。原科学门未通过，阶段5尚未进入。'
path.write_text('\n'.join(lines)+'\n'+paragraph,encoding='utf-8',newline='\n')
path=ROOT/'README.md';s=path.read_text(encoding='utf-8');_,tail=s.split('\n---\n',1)
head='''# N01 当前阶段：阶段4随机精度诊断闭合，科学门未过

> 当前执行状态（2026-10-20）：三轮REAL校准、五项固定方向诊断和随机精度诊断均已完成并审核。原最终科学门及两头成对抽样精度门未通过，当前0科学作业、0定时；REAL响应匹配尚未建立，阶段5待验。

[最近独立审核](receipts/joint_response_01_20261020/random_precision_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)完成864差分列、36固定检查、162风险视图和8库2097152潜变量的原环境精确重放。正负成对抽样的J离散度比为0.67150/0.87855，native wave/QPSD为0.86858/0.80620；两头native和第二头J未达到≤0.75，故未通过预登记门，零新候选或更新。

两头平均绝对native预测误差虽约下降10%/8%，仍须用保存重复库区分随机波动与局部模型的系统误差。三轮REAL校准的两神经各24/96最终改善门失败，固定中心三候选曲率序列已停止，历史失败保持。

作业于12:00:56（UTC+8）正常结束，1764.60秒，879源码与53输出哈希核验通过。本次12:29单次定时已删除，见[消费回执](receipts/joint_response_01_20261020/random_precision_diagnostic_01/WAKE1229_AUTOMATION_CONSUMED.json)；没有等待事项时不保留定时。

以下保留历史快照；当前状态以本节、三台账及对应回执为准。

'''
path.write_text(head+'\n---\n'+tail,encoding='utf-8',newline='\n')
(HERE/'CLOSURE_RECEIPT.json').write_text(json.dumps(dict(status='CLOSED_AUDITED_PRECISION_NEGATIVE',UTC=now,prior_138_jobs_unchanged=True,active_jobs=0,automations=0,scientific_acceptance=False,terminal_review=rel+'/terminal_01/TERMINAL_REVIEW.json'),indent=2)+'\n',encoding='utf-8')
print('Closed job 139; zero active science and zero scheduled wakes.')
