from pathlib import Path
from datetime import datetime, timezone
import copy, json, shutil

root=Path('E:/GAN/research/N01'); here=Path(__file__).resolve().parent
review=json.loads((here/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
raw=json.loads((here/'WAKE0904_STATUS_RAW.json').read_bytes())
assert review['two_seed_FRESH_pass'] and review['audit_counts']['independent_joint_solutions']==3
assert review['audit_counts']['secant_corrections_recomputed']==1
assert raw['processes']==[] and raw['files']['control_01/exit_receipt.json']['exit_code']==0
assert not Path('C:/Users/10629/.codex/automations/n01/automation.toml').exists()
before=here/'before_close_wake0904'; before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name,before/name)
now=datetime.now(timezone.utc).isoformat()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes())
    if name=='jobs.json':
        old=copy.deepcopy(d['jobs']); assert len(old)==132
        assert d['jobs'][-1]['job_id']=='JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_FIX01_20261020'
        d['jobs'][-1].update(status='completed_audited_local_gates_pass',native_exit_code=0,
            ended_UTC=review['native_exit']['ended_utc'],execution_counts=review['execution_counts'],
            terminal_review=(here/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),
            two_seed_FRESH_pass=True,parameter_updates=0,scientific_acceptance=False)
        assert d['jobs'][:-1]==old[:-1]
    d.update(status='JOINT_RESPONSE_01_original_draw_precision_fix01_closed',updated_UTC=now,
        active_scientific_jobs=0,active_job_id=None,active_run_id=None,pending_job_id=None,
        training_pid=None,supervisor_pid=None,current_job_native_exit_code=0,
        current_job_ended_UTC=review['native_exit']['ended_utc'],last_verified_UTC=raw['UTC'],
        execution_state_asof_UTC=raw['UTC'],scientific_acceptance=False,
        current_action='Audited two-head 32-draw local directions; preparing a minimum full-coverage calibration with matched physical controls.',
        next_action='Register three bounded full-coverage outer iterations per track; retain final gates and kernel controls.',
        followup=dict(status='DELETED_CONFIRMED',automation_id=None,count=0),followup_status='DELETED_CONFIRMED',
        followup_automation_id=None,followup_automation_status='DELETED_CONFIRMED',followup_scheduled_local=None,
        followup_next_run_local=None,next_wake_local=None)
    d['joint_response_01'].update(original_draw_precision_fix01_status='completed_audited_local_gates_pass',
        original_draw_precision_fix01_completed=True,original_draw_precision_fix01_execution_counts=review['execution_counts'],
        original_draw_precision_fix01_terminal_review=(here/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),
        original_draw_precision_fix01_two_seed_FRESH_pass=True)
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
paragraph='''
## 2026-10-20 09:04接续：原生成器32draw局部门通过

[FIX01终止审核](receipts/joint_response_01_20261020/original_draw_precision_fix01/terminal_01/TERMINAL_REVIEW.json)确认原生exit 0、565.78秒、RSS峰5,857,550,336字节；815源文件、35锁定输入、13输出及全部哈希通过。108差分列、3完整联合解、3候选、1割线修正和2次FRESH检查独立复算，求解参数最大差2.09e-17；没有零方向分支或未审计求解。实际688生成调用22,192,384行、2058观察调用66,576,384行、native 75,005,952行，零提交更新。

两头第1/第2候选通过全部FIT/CHECK，FRESH联合风险分别剩余0.918138/0.967731，最大native wave比1.002091/1.008556；QPSD比0.98557—0.99121，noise比1。局部保护通过不等于最终5%原生距离改善。旧4draw候选在同32draw FRESH上也通过，联合风险剩余0.926661/0.985056；本次不能声称只有32draw可行，也不能由这两个方向证明梯度全面更准确。

所有SOURCE/REAL行、原4draw中心、原32draw五中心九列和完整跨模块矩阵已核验。FRESH复用历史拟合角色行及投影，只有新增28draw随机流；保存向量审核未独立再生成完整波形及RMS/flag统计。09:04定时已消费删除，当前无等待事项。下一项前瞻登记三轮完整拟合角色覆盖的校准及匹配物理基线，全部旧失败、原最终门与阶段5保持。
'''
for name in ('README.md','JOINT_RESPONSE_01_PLAN.md'):
    path=root/name; lines=path.read_text('utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('> 当前'):
            lines[i]='> 当前执行状态（2026-10-20）：原生成器32draw局部诊断已完成并独立审核，两头FIT/CHECK/FRESH通过，零参数更新；准备最小完整覆盖校准及匹配物理对照。当前0科学作业、0定时；REAL响应匹配尚未建立。'; break
    text='\n'.join(lines)+'\n'
    if name=='JOINT_RESPONSE_01_PLAN.md': text+=paragraph
    path.write_text(text,encoding='utf-8',newline='\r\n')
print(json.dumps({'status':'CLOSED_AUDITED','active_jobs':0,'automations':0,'prior_131_jobs_unchanged':True}))
