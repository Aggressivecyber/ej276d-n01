from datetime import datetime,timezone
from pathlib import Path
import copy,json,shutil
root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
r=json.loads((here/'terminal_01/TERMINAL_REVIEW.json').read_bytes());m=json.loads((here/'terminal_01/REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert r['status']=='PASS_SAVED_NATIVE_CURVATURE_DIAGNOSTIC_AUDIT' and not r['two_seed_curvature_all_local_pass'] and not m['processes']
before=here/'before_close';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/name,before/name)
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 p=root/name;d=json.loads(p.read_bytes())
 if name=='jobs.json':
  old=copy.deepcopy(d['jobs']);assert len(old)==134 and old[-1]['job_id']=='JOINT_RESPONSE_01_NATIVE_CURVATURE_DIAGNOSTIC_20261020'
  d['jobs'][-1].update(status='completed_audited_local_gates_failed',native_exit_code=0,ended_UTC=r['native_exit']['ended_utc'],execution_counts=r['execution_counts'],
      terminal_review=(here/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),two_seed_local_pass=False,parameter_updates=0,scientific_acceptance=False)
  assert d['jobs'][:-1]==old[:-1]
 d.update(status='JOINT_RESPONSE_01_native_curvature_diagnostic_closed_negative',updated_UTC=datetime.now(timezone.utc).isoformat(),active_scientific_jobs=0,
    active_job_id=None,active_run_id=None,pending_job_id=None,training_pid=None,supervisor_pid=None,current_job_native_exit_code=0,current_job_ended_UTC=r['native_exit']['ended_utc'],
    last_verified_UTC=m['UTC'],execution_state_asof_UTC=m['UTC'],scientific_acceptance=False,
    current_action='Fixed curvature candidate improved the native model but failed seed1 initial-anchor guards; saved native-gain feasibility diagnosed.',
    next_action='Test one FIT-derived common-native-improvement candidate per head with fixed controls and new FRESH data; no updates or training.',
    followup=dict(status='NOT_NEEDED_NO_LONG_WAIT',automation_id=None,count=0),followup_status='NOT_NEEDED_NO_LONG_WAIT',followup_automation_id=None,
    followup_automation_status='NOT_NEEDED_NO_LONG_WAIT',followup_scheduled_local=None,followup_next_run_local=None,next_wake_local=None)
 d['joint_response_01'].update(native_curvature_diagnostic_status='completed_audited_local_gates_failed',native_curvature_diagnostic_completed=True,
    native_curvature_diagnostic_execution_counts=r['execution_counts'],native_curvature_diagnostic_terminal_review=(here/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),native_curvature_two_seed_pass=False)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
paragraph='''
## 2026-10-20 保存曲率诊断闭合：减少退化，但两头局部门未全过

[保存矩阵诊断](receipts/joint_response_01_20261020/original_precision_REAL_pilot_01/SAVED_CONSTRAINT_DIAGNOSIS.json)确认6个原首神经候选均预测FIT native wave均值下降而实际上升。保持完整联合目标，加入原±0.004差分的正对角native曲率和固定初值锚后，分别固定一个新候选与原线性对照，[原生小试验及独立复算](receipts/joint_response_01_20261020/native_curvature_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)完成：exit0、38.64秒，828源文件/31pins/13输出，40生成调用1,310,720行；2完整候选解、20接受检查、90风险视图通过，零更新。

seed0新候选全部五库通过；seed1 FIT0初值QPSD最大1.010034、新FRESH初值wave最大1.010526，超过1.01，按原门判失败。FRESH联合剩余0.90157/0.87933，不能据此启动训练。对角曲率不是可靠上界，原最终5%native改善仍未得到支持。

[原生共同增益可行性](receipts/joint_response_01_20261020/native_curvature_diagnostic_01/SAVED_NATIVE_GAIN_DIAGNOSIS.json)只使用已保存FIT矩阵，在每个FIT至少1%联合改善的约束下，正对角局部模型预测可同时降低所有wave/QPSD约0.556%/0.652%。加入FIT候选实测正误差的秩一曲率对照中，seed0在原200步求解限额未收敛，未使用其参数；不把它判成物理不可行。下一项仅检验两头均完成的原正对角共同增益方向，原曲率候选作固定对照，新FRESH仅评价，不自动训练。当前无长等待，未设定时。
'''
for name in ('README.md','JOINT_RESPONSE_01_PLAN.md'):
 path=root/name;lines=path.read_text('utf-8').splitlines()
 for i,line in enumerate(lines):
  if line.startswith('> 当前'):
   lines[i]='> 当前执行状态（2026-10-20）：三轮REAL校准与曲率诊断均已闭合审核，科学及两头局部门仍未全过。正在检验原生wave/QPSD共同改善方向；当前0科学作业、0定时，REAL响应匹配尚未建立。';break
 content='\n'.join(lines)+'\n'
 if name=='JOINT_RESPONSE_01_PLAN.md':content+=paragraph
 path.write_text(content,encoding='utf-8',newline='\r\n')
print(json.dumps(dict(status='CLOSED_AUDITED_NEGATIVE',prior_133_jobs_unchanged=True,active_jobs=0,automations=0)))
