from datetime import datetime,timezone
from pathlib import Path
import hashlib,json

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
result=json.loads((HERE/'SOLVE_RESULT.json').read_bytes());audit=json.loads((HERE/'INDEPENDENT_AUDIT.json').read_bytes())
exit_record=json.loads((HERE/'EXIT_RECEIPT.json').read_bytes())
assert exit_record['exit_code']==0 and audit['two_seed_local_feasible']
assert audit['status']=='PASS_INDEPENDENT_FRESH_RESTORATION_MATRIX_AND_CERTIFICATE_AUDIT'
assert result['execution_counts']['local_solves']==2 and result['execution_counts']['generator_calls']==0
assert result['peak_rss_bytes']<=2*1024**3 and result['elapsed_seconds']<=300
now=datetime.now(timezone.utc).isoformat();rel=HERE.relative_to(ROOT).as_posix()
for filename in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    path=ROOT/filename;d=json.loads(path.read_bytes())
    assert d['active_job_id']=='JOINT_RESPONSE_01_RECENTERED_RESTORATION_SOLVES_FIX01_20261020'
    d.update(status='JOINT_RESPONSE_01_fresh_restoration_KKT_audited',updated_UTC=now,
        execution_state_asof_UTC=audit['UTC'],active_scientific_jobs=0,active_job_id=None,pending_registered_job_id=None,
        current_action='Both fresh-center fixed restoration programs solved once and independently KKT-certified; correction norms0.00207434534/0.00144677842 within0.003125. No actual generator candidate or update executed.',
        next_action='Prepare and register the fixed two-candidate actual hard-readout check, retaining original joint/native/trust gates; no extra solve or candidate search.',
        active_scientific_jobs_status_basis='Local solve process exited0 and independent saved matrix/KKT audit completed.',scientific_acceptance=False)
    d['joint_response_01'].update(recentered_restoration_status='completed_two_seed_KKT_audited',
        recentered_restoration_result=rel+'/SOLVE_RESULT.json',recentered_restoration_audit=rel+'/INDEPENDENT_AUDIT.json',
        recentered_restoration_solver_invocations=2,recentered_restoration_actual_candidates_evaluated=0,
        recentered_restoration_generator_calls=0,recentered_restoration_parameter_updates=0,
        recentered_restoration_correction_norms=[r['correction_norm'] for r in audit['reports']],
        current_scientific_acceptance=False)
    if filename=='jobs.json':
        assert len(d['jobs'])==144
        d['jobs'][-1].update(status='completed_two_seed_KKT_audited',verified_UTC=now,native_exit_code=0,
            result=rel+'/SOLVE_RESULT.json',independent_audit=rel+'/INDEPENDENT_AUDIT.json',
            execution_counts=result['execution_counts'],elapsed_seconds=result['elapsed_seconds'],
            peak_rss_bytes=result['peak_rss_bytes'],two_seed_local_feasible=True,scientific_acceptance=False)
    path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
header='''# N01 当前阶段：新中心恢复求解已通过独立KKT审核

> 2026-10-20 16:28（UTC+8）：742个唯一导数列已独立审核；两头各一次恢复求解和全部约束/KKT通过，修正范数0.002074/0.001447，半径0.003125。当前0科学作业、0定时。下一项直接登记并执行固定候选硬读出检查。阶段4科学验收未过，阶段5未进入。

[合并导数审核](receipts/joint_response_01_20261020/recentered_derivative_completion_01/terminal_01/TERMINAL_REVIEW.json)复算308数组、71680条角色身份、742列、84个风险视图和14个控制门；192源/25输出、计数和资源界均通过。job142退出0，新增424列与旧318列合并；原job141中断失败及9列/20生成重复成本保留。

[固定恢复独立审核](receipts/joint_response_01_20261020/recentered_restoration_solve_fix01/INDEPENDENT_AUDIT.json)重建两组10709×53约束。两头驻点残差约1.60e-15/5.65e-15，互补约6.98e-11/5.39e-11；仅两次局部求解，47.64秒、峰RSS135512064字节，生成/实际候选/参数更新均零。全部七库均已参与拟合约束，局部可行性不代表独立物理验证。

首次求解进程在调用求解器前触发错误的噪声绝对零检查，job143保留为零求解的工程失败。最小fix01只要求噪声改变量零，精确保留原约1e-20的坐标；旧源码、失败回执与配置保留，半径、余量和科学门限未改。

按用户指令，本聊天只推进N01研究；NAS挂载、连接和传输管理交由另一agent。科学文件使用现有路径，矩阵证据位于NAS；当前没有必须等待的科学进程，因此没有定时。以下历史快照保留，当前以本节、三台账与回执为准。

'''
path=ROOT/'README.md';old=path.read_text(encoding='utf-8')
path.write_text(header+'---\n'+old.split('---\n',1)[1],encoding='utf-8',newline='\n')
path=ROOT/'JOINT_RESPONSE_01_PLAN.md';lines=path.read_text(encoding='utf-8').splitlines()
lines[2]='> 最后核验（2026-10-20 16:28，UTC+8）：742列导数审核通过，两头各一次新中心恢复求解及独立KKT通过；实际候选尚未执行。0科学作业、0定时。立即登记固定候选硬读出检查；阶段4科学验收未过，阶段5未进入。'
text='\n'.join(lines)+'\n\n## 16:28 新中心恢复求解与独立证书\n\n'+header.split('\n\n',2)[2]
path.write_text(text,encoding='utf-8',newline='\n')
receipt=dict(status='PASS_TWO_FIXED_RESTORATION_SOLVES_CLOSED',UTC=now,active_scientific_jobs=0,followup_count=0,
    solver_invocations=2,actual_candidate_evaluations=0,scientific_acceptance=False,
    audit_sha256=hashlib.sha256((HERE/'INDEPENDENT_AUDIT.json').read_bytes()).hexdigest())
(HERE/'CLOSURE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt))
