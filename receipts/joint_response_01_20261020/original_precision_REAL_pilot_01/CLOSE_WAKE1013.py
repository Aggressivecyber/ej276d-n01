from pathlib import Path
from datetime import datetime, timezone
import copy, json, shutil

root=Path('E:/GAN/research/N01'); here=Path(__file__).resolve().parent
review=json.loads((here/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
raw=json.loads((here/'WAKE1013_STATUS_RAW.json').read_bytes())
assert review['status']=='PASS_SAVED_ORIGINAL_PRECISION_REAL_PILOT_AUDIT'
assert review['final_gates_recomputed']==192 and not review['REAL_calibration_candidate_pass_all']
assert raw['processes']==[] and raw['files']['control_01/exit_receipt.json']['exit_code']==0
assert not Path('C:/Users/10629/.codex/automations/n01-real/automation.toml').exists()
before=here/'before_close_wake1013'; before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name,before/name)
now=datetime.now(timezone.utc).isoformat()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes())
    if name=='jobs.json':
        old=copy.deepcopy(d['jobs']); assert len(old)==133
        assert old[-1]['job_id']=='JOINT_RESPONSE_01_ORIGINAL_PRECISION_REAL_PILOT_20261020'
        d['jobs'][-1].update(status='completed_audited_scientific_gates_failed',native_exit_code=0,
            ended_UTC=review['native_exit']['ended_utc'],execution_counts=review['execution_counts'],
            terminal_review=(here/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),
            failed_final_gates_by_seed={s:len(v['failed_gates']) for s,v in review['final_evidence'].items()},
            scientific_acceptance=False)
        assert d['jobs'][:-1]==old[:-1]
    d.update(status='JOINT_RESPONSE_01_original_precision_REAL_pilot_closed_negative',updated_UTC=now,
        active_scientific_jobs=0,active_job_id=None,active_run_id=None,pending_job_id=None,
        training_pid=None,supervisor_pid=None,current_job_native_exit_code=0,
        current_job_ended_UTC=review['native_exit']['ended_utc'],last_verified_UTC=raw['UTC'],
        execution_state_asof_UTC=raw['UTC'],scientific_acceptance=False,
        current_action='Audited four complete tracks; both neural endpoints fail 24 of 96 original final gates. Diagnosing saved native prediction error and active constraints.',
        next_action='Finish saved-matrix diagnosis before registering a bounded zero-update method test; do not extend the three-round run.',
        followup=dict(status='DELETED_CONFIRMED',automation_id=None,count=0),followup_status='DELETED_CONFIRMED',
        followup_automation_id=None,followup_automation_status='DELETED_CONFIRMED',followup_scheduled_local=None,
        followup_next_run_local=None,next_wake_local=None)
    d['joint_response_01'].update(original_precision_REAL_pilot_status='completed_audited_scientific_gates_failed',
        original_precision_REAL_pilot_completed=True,original_precision_REAL_pilot_execution_counts=review['execution_counts'],
        original_precision_REAL_pilot_terminal_review=(here/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),
        original_precision_REAL_pilot_final_pass=False,original_precision_REAL_pilot_failed_final_gates_by_seed={'0':24,'1':24})
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
paragraph='''
## 2026-10-20 10:13接续：三轮完整覆盖校准闭合，最终改善门失败

[终止独立审核](receipts/joint_response_01_20261020/original_precision_REAL_pilot_01/terminal_01/TERMINAL_REVIEW.json)确认exit 0、3174.69秒、819源文件/23锁定输入、51输出105,638,165字节全部哈希通过。600差分列、20联合求解、8割线、80接受检查及192最终门全部复算，独立求解最大参数差1.98e-17。四轨迹均覆盖6465 SOURCE拟合父/18948 REAL行；神经各2次、物理各3次更新。10:13定时已消费删除，当前无活动科学作业。

两神经各24/96最终改善门失败。seed0联合风险剩余0.89263—0.90543、peak0.91416—0.92798、native wave0.98866—0.99125、QPSD0.99400—1.00676；seed1分别0.89510—0.90834、0.92256—0.92760、0.98859—0.99501、0.99682—1.00475。原noise/macro/flag门通过。物理末端native wave约2.47—2.50 M0、QPSD约8.85—9.36 M0，不能将其相对自身起点的改善当作成功响应。

两库同父128×1024的保存均值、协方差和方差已独立重算；仍无REAL条件核真值。大步候选线性native预测通过而实际wave退化，缩步后增益不足；下一项先使用完整保存矩阵定位曲率、固定初值锚及拟合/检查差异，不增加本次轮数或扩大网络。原模型、门限、历史负结果和阶段5待验状态保持。
'''
for name in ('README.md','JOINT_RESPONSE_01_PLAN.md'):
    path=root/name; lines=path.read_text('utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('> 当前'):
            lines[i]='> 当前执行状态（2026-10-20）：32draw三轮REAL校准已结束并独立审核；两神经各2次更新，各24/96最终改善门失败。正在诊断保存矩阵中的native预测误差与约束；当前0科学作业、0定时，REAL响应匹配尚未建立。';break
    content='\n'.join(lines)+'\n'
    if name=='JOINT_RESPONSE_01_PLAN.md':content+=paragraph
    path.write_text(content,encoding='utf-8',newline='\r\n')
print(json.dumps(dict(status='CLOSED_AUDITED_NEGATIVE',active_jobs=0,automations=0,prior_132_jobs_unchanged=True)))
