from datetime import datetime,timezone
from pathlib import Path
import copy,json,shutil

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
review=json.loads((here/'terminal_01/TERMINAL_REVIEW.json').read_bytes());terminal=json.loads((here/'terminal_01/REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status']=='PASS_SAVED_CURVATURE_UPDATE_DIAGNOSTIC_AUDIT' and not review['two_seed_curvature_update_all_local_pass'] and not terminal['processes']
before=here/'before_close';before.mkdir();now=datetime.now(timezone.utc).isoformat()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/name,before/name)
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 d=json.loads((root/name).read_bytes())
 if name=='jobs.json':
  old=copy.deepcopy(d['jobs']);assert len(old)==137 and old[-1]['job_id']=='JOINT_RESPONSE_01_CURVATURE_UPDATE_DIAGNOSTIC_20261020'
  d['jobs'][-1].update(status='completed_audited_local_gates_failed',native_exit_code=0,ended_UTC=review['native_exit']['ended_utc'],execution_counts=review['execution_counts'],terminal_review=(here/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),two_seed_local_pass=False,parameter_updates=0)
  assert d['jobs'][:-1]==old[:-1]
 d.update(status='JOINT_RESPONSE_01_curvature_update_diagnostic_closed_negative',active_scientific_jobs=0,active_job_id=None,active_run_id=None,pending_job_id=None,training_pid=None,supervisor_pid=None,current_job_native_exit_code=0,current_job_ended_UTC=review['native_exit']['ended_utc'],last_verified_UTC=terminal['UTC'],execution_state_asof_UTC=terminal['UTC'],updated_UTC=now,current_action='Fixed mixed-curvature candidate audited negative: seed1 FIT0 QPSD anchor exceeds 1.01.',next_action='Diagnose the measured FIT quadratic residual from saved vectors before any new prospective method check.')
 d['joint_response_01'].update(curvature_update_diagnostic_status='completed_audited_local_gates_failed',curvature_update_diagnostic_completed=True,curvature_update_diagnostic_terminal_review=(here/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),curvature_update_diagnostic_execution_counts=review['execution_counts'])
 (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
(here/'CLOSURE_RECEIPT.json').write_text(json.dumps(dict(status='CLOSED_AUDITED_NEGATIVE',UTC=now,prior_136_jobs_unchanged=True,active_jobs=0,automations=0,scientific_acceptance=False),indent=2)+'\n',encoding='utf-8')
paragraph='''
## 2026-10-20 一次混合曲率修正闭合：一个FIT初值锚仍失败

[独立审核](receipts/joint_response_01_20261020/curvature_update_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)确认858源/61pins/13输出，exit0、73.68秒、40生成调用、零更新；原36宏块/27native约束、两头锥问题原始及对偶证书、20接受检查和90风险视图全部复算。seed0五库全过；seed1仅FIT0初值QPSD最大1.010555超过1.01，保持失败。两头新FRESH joint剩余0.966919/0.939009，当前native wave最大0.997678/1.000099、QPSD最大1.005895/0.995357；原最终科学门仍未通过。

本次新FRESH流无前三项跨头复用。下一步只用FIT保存误差检查混合曲率模型，未提交端点、未加校准轮数；当前0作业、0定时。
'''
# Use the audited FRESH maxima instead of hand-entered summary values.
for seed in (0,1):
 entry=review['summaries'][str(seed)]['by_model']['curvature_update']['FRESH']
 if seed==0:
  paragraph=paragraph.replace('0.997678/1.000099',f"{entry['native_ratio_max'][0]:.6f}/1.000099").replace('1.005895/0.995357',f"{entry['native_ratio_max'][1]:.6f}/0.995357")
plan=root/'JOINT_RESPONSE_01_PLAN.md';text=plan.read_text('utf-8');lines=text.splitlines();lines[2]='> 当前执行状态（2026-10-20）：三轮REAL校准及四项后续固定方向诊断已闭合审核，科学及两头局部门未全过；正在分析剩余FIT曲率误差。当前0科学作业、0定时，REAL响应匹配尚未建立。';plan.write_text('\n'.join(lines)+'\n'+paragraph,encoding='utf-8',newline='\r\n')
readme=root/'README.md';text=readme.read_text('utf-8').replace('新固定方向已正确启动，当前1科学作业、0定时','新固定方向已闭合审核，仍有一个FIT初值QPSD锚失败；当前0科学作业、0定时').replace('三项后续诊断','四项后续诊断').replace('下一步执行原硬读出固定候选检查，零更新','实际原硬读出检查未全过，下一步分析保存FIT曲率误差，零更新');readme.write_text(text,encoding='utf-8',newline='\r\n')
print('Closed audited negative; no active science or followup automation.')
