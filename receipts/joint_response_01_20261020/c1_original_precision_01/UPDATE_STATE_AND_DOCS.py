from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, shutil

root = Path('E:/GAN/research/N01')
here = Path(__file__).resolve().parent
raw = json.loads((here/'WAIT_ESTIMATE_RAW.json').read_bytes())
estimate = json.loads((here/'WAIT_ESTIMATE.json').read_bytes())
automation = json.loads((here/'AUTOMATION_CREATE_RECEIPT.json').read_bytes())
assert automation['status'] == 'PASS_SINGLE_WAKE_CREATED_AND_CONFIG_VERIFIED' and automation['count'] == 1
progress = raw['files']['run_01/PROGRESS.json']
counts = progress['execution_counts']
assert counts['finite_difference_columns'] == 54 and counts['joint_candidate_trials'] == 2 and counts['teacher_accepted_joint_updates'] == 1
now = datetime.now(timezone.utc).isoformat()
prefix = 'receipts/joint_response_01_20261020/c1_original_precision_01/'
old_prefix = 'receipts/joint_response_01_20261020/c1_precision_pilot_01/'
(here/'before_current_docs').mkdir()
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json', 'README.md', 'JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name, here/'before_current_docs'/name)
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root/name).read_bytes())
    assert d['active_job_id'] == 'JOINT_RESPONSE_01_C1_ORIGINAL_PRECISION_20261020' and d['active_scientific_jobs'] == 1
    d.update(status='JOINT_RESPONSE_01_C1_original_precision_running_single_wake', updated_UTC=now,
        execution_state_asof_UTC=raw['UTC'], current_stage='3_C1_original_precision_confirmation',
        active_work_package='3_C1_original_precision_confirmation',
        current_action='Positive control independently passed all original gates. Direct original-teacher recovery first iteration completed with one accepted update from two candidates.',
        next_action='At 08:15 consume one wake, verify terminal identity and audit direct original-teacher recovery. Continue ready work; synthetic recovery does not establish REAL acceptance.',
        followup_status='ACTIVE_SINGLE_WAKE', followup_automation_id='n01', followup_scheduled_local=automation['next_wake_local'],
        followup_next_run_local=automation['next_wake_local'], next_wake_local=automation['next_wake_local'],
        followup=dict(status='ACTIVE', kind='heartbeat', automation_id='n01', count=1, next_wake_local=automation['next_wake_local'],
            estimate_receipt=prefix+'WAIT_ESTIMATE.json', verification_receipt=prefix+'AUTOMATION_CREATE_RECEIPT.json'),
        scientific_acceptance=False)
    d['joint_response_01'].update(c1_precision_pilot_status='completed_audited_positive_control_all_gates_pass',
        c1_precision_pilot_terminal_review=old_prefix+'terminal_01/TERMINAL_REVIEW.json',
        c1_precision_pilot_matched_recovery_pass=True, c1_precision_pilot_original_target_report_pass=True,
        c1_original_precision_status='running_first_joint_update_verified',
        c1_original_precision_execution_counts_last_verified=counts, c1_original_precision_status_asof_UTC=raw['UTC'],
        c1_original_precision_estimated_finish_local=estimate['estimated_finish_local'],
        c1_original_precision_scientific_acceptance=False, c1_original_precision_automatic_REAL_promotion=False)
    if name == 'jobs.json':
        assert d['jobs'][-2]['status'] == 'completed_audited_positive_control_all_gates_pass'
        d['jobs'][-1].update(status='running_first_joint_update_verified', status_asof_UTC=raw['UTC'],
            last_verified_execution_counts=counts,
            native_self_peak_RSS_bytes=raw['files']['run_01/seed0_joint_MEMORY_ITER00.json']['peak_RSS_bytes'],
            last_verified_supervisor_peak_RSS_bytes=raw['files']['control_01/resource_status.json']['peak_rss_bytes'],
            followup_local=automation['next_wake_local'], followup_automation_id='n01')
    (root/name).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\r\n')

current = '''# N01 当前阶段：高精度正对照通过，原教师直接恢复在运行

[32 draw有界正对照已完成并独立复算](receipts/joint_response_01_20261020/c1_precision_pilot_01/terminal_01/TERMINAL_REVIEW.json)：8个联合方向、11个候选、7次接受更新；同家族C1和原教师报告各36/36硬观察门、18/18条件精度门通过。原生exit 0，耗时1012.74秒，RSS峰6.22GiB；780源文件、120锁定输入、57输出核验。均值与随机参数在同一完整硬读出目标中共同变化，原门限保持。

新正对照对原教师报告的seed0两库Q方差比分别0.65102/0.66156，seed1为0.53514/0.54170；此前约0.494及0.4895失败分别保留。该端点拟合的是C1教师，原教师报告通过尚不能替代直接拟合原教师的恢复确认。保存差分、联合解、全部实际候选风险／完整信赖向量、参数链、父覆盖及条件矩均已复算；最终native门和flag计数使用保存汇总，未重新生成原始波形。

已登记并正确启动[原教师32 draw直接恢复确认](docs/JOINT_RESPONSE_01_C1_ORIGINAL_PRECISION_20261020.md)：从原跨家族完整父集的两个ITER11冻结端点出发，直接拟合原JointResponse01教师。学生仍C1，两头各固定4轮，直接复用已通过正对照的联合求解函数。全部6465拟合父×32 draw、原18特征因子、54坐标、差分／求解／接受规则不变；128报告父不拟合。FIT、CHECK和末端报告使用分离的新随机库。4轮全部结束后才评价原36硬门及18条件精度门，不按中间Q方差选择端点或追加轮数。

[原生启动核验](receipts/joint_response_01_20261020/c1_original_precision_01/STARTUP_ASSESSMENT.json)确认07:55:25派发、785源文件／125锁定输入、科学PID137691身份一致。截至07:57:41，第1/8个联合方向完成：54/432差分列、2候选、1接受更新，9个原特征因子精确重现；该时刻之后的进度未核验。第一轮进程RSS峰6,536,024,064字节（约6.09GiB）。预算4CPU、GPU分配2.75GiB、RSS12GiB、内部1500秒／外部1740秒加20秒，输出192MiB；不自动重启或延长。

07:48旧定时已消费并删除。按首轮121.53秒、剩余7轮、原教师末端报告每头约9.46秒及输出余量，[预计08:12:31完成](receipts/joint_response_01_20261020/c1_original_precision_01/WAIT_ESTIMATE.json)。已核验 **2026-10-20 08:15（UTC+8）本聊天单次接续**，留约149秒余量，见[定时回执](receipts/joint_response_01_20261020/c1_original_precision_01/AUTOMATION_CREATE_RECEIPT.json)。醒来先收集与复算实际结果，继续所有当前可执行工作。

原物理基线与固定响应人口压力对照保留；原REAL两头各24/96改善门失败不变。合成恢复、REAL联合改善与阶段5分别判定，科学响应匹配仍未建立。已暴露A0/M0及128报告父仅支持冻结系统后验评价，全流程未见验证尚未完成。

'''
readme = (root/'README.md').read_text(encoding='utf-8')
boundary = readme.index('以下保留历史快照；当前状态以本节、三台账及对应回执为准。')
(root/'README.md').write_text(current+readme[boundary:], encoding='utf-8', newline='\r\n')
plan = (root/'JOINT_RESPONSE_01_PLAN.md').read_text(encoding='utf-8')
lines = plan.splitlines()
assert lines[2].startswith('> 当前执行状态')
lines[2] = '> 当前执行状态（2026-10-20）：32draw正对照已独立复算，两教师报告均36/36硬门与18/18条件精度门通过。原教师直接恢复已从原跨家族端点启动，首轮2候选1接受；08:15本聊天单次接续。原失败及阶段4/5未完成状态保持。'
appendix = '''

## 2026-10-20 07:48接续：正对照通过，直接拟合原教师

[正对照终止复算](receipts/joint_response_01_20261020/c1_precision_pilot_01/terminal_01/TERMINAL_REVIEW.json)确认780源文件、120锁定输入、57输出及全部哈希，432差分列、8联合方向、11候选、7接受更新。两家族各36/36硬门、18/18精度门通过；原教师报告Q方差比seed0为0.65102/0.66156、seed1为0.53514/0.54170。原生exit 0、1012.74秒、RSS峰6.22GiB。原4draw失败与资源停止保持；此结果拟合C1教师，尚需直接原教师恢复。

已单独登记[原教师高精度确认](docs/JOINT_RESPONSE_01_C1_ORIGINAL_PRECISION_20261020.md)，配置SHA256 `75f5d9b100cf4387f7409d218f753c18dac2ead658618bcd89a28c595fcd79ff`。从原跨家族两头ITER11端点出发，每头固定4轮，同一fit_track_precision函数；完整6465父×32draw、原18特征因子、54坐标及全部规则保持。新拟合／报告随机流互斥，末端只对原教师报告原36硬门与18精度门；128报告父排除在拟合之外，不按报告真值调参数或追加迭代。

07:57:41快照核验785源／125pins、boot及PID/ticks，第一轮54列、1方向、2候选／1接受已完成；进程RSS峰约6.09GiB。首轮121.53秒，按余下7轮最多3候选及原实测末端报告估计08:12:31完成；08:15单次苏醒已核验，约149秒余量。内部08:20:25／外部最晚08:24:45边界保持。收集与独立复算脚本已准备并通过AST，终止后才执行；原物理／人口对照、REAL失败及科学接受边界不变。
'''
(root/'JOINT_RESPONSE_01_PLAN.md').write_text('\n'.join(lines).rstrip()+appendix, encoding='utf-8', newline='\r\n')
print(json.dumps(dict(status='UPDATED_THREE_LEDGERS_AND_CURRENT_DOCS', UTC=now,
    science_asof_UTC=raw['UTC'], single_wake_local=automation['next_wake_local'], scientific_acceptance=False)))
