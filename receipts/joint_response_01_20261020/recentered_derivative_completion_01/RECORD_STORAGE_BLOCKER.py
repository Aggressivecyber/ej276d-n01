from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
PREV = HERE.parent/'recentered_derivative_diagnostic_01'
now = datetime.now(timezone.utc).isoformat()
local = datetime.now(timezone(timedelta(hours=8)))
review = json.loads((HERE/'LOCAL_PREPARATION_REVIEW.json').read_bytes())
stopped = json.loads((HERE/'FINAL_DEPLOYMENT_STOP_RECEIPT.json').read_bytes())
receipt_reads = [json.loads(line) for line in (HERE/'THIN_DEPLOYMENT_RECEIPT_READ_STDOUT.jsonl').read_text(encoding='utf-8').splitlines()]
assert stopped['status'] == 'OWN_DEPLOYMENT_HELPERS_STOPPED'
assert receipt_reads[-1]['status'] == 'DEPLOYMENT_RECEIPT_NOT_FOUND'
assert review['status'] == 'PASS_LOCAL_PREPARATION_ONLY_NAS_GATE_BLOCKED' and not review['dispatch_ready']
assert not Path('C:/Users/10629/.codex/automations/n01/automation.toml').exists()
receipt = dict(status='WAITING_EXTERNAL_NAS_ACCESS_COMPLETION_NOT_STARTED', UTC=now,
    original_job_id='JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_DIAGNOSTIC_20261020',
    original_job_failed=True, saved_FD_columns_verified=318, executed_but_unsaved_FD_columns=9,
    unexecuted_original_FD_columns=415, fixed_completion_FD_columns=424,
    prepared_job_id='JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_COMPLETION_20261020',
    prepared_config='configs/joint_response_01_recentered_derivative_completion_thin_20261020.json',
    prepared_config_sha256='e5727dcb8356a2212a6d1c05fe0e2ad471dab60661c4c97f69ffb8c355c10ce1',
    local_source_files_verified=192, source_pins=191, external_prefix_files=12,
    local_text_files=7, external_immutable_sources=185,
    fullcopy_deployment='180s local SSH wait timed out; owned PID215696 stopped after identity verification.',
    partial_copy_repair='180s local SSH wait timed out; owned PID217689 stopped after identity verification.',
    thin_deployment='Remote session ended with local SSH exit4294967295 after startup marker; no native deployment pass received.',
    thin_start_UTC='2026-10-20T06:42:10.922262+00:00', thin_pid=220732,
    thin_deployment_receipt_present=False, receipt_absence_verified_UTC=receipt_reads[-1]['UTC'],
    original_and_deployment_processes_absent=stopped,
    new_scientific_calls=0, completion_dispatched=False, active_scientific_jobs=0,
    automatic_restart=False, new_training_updates=0, restoration_solves=0,
    scientific_acceptance=False, followup_count=0,
    no_timer_reason='No scientific computation or deployment remains running; NAS recovery has no verified completion estimate.',
    required_external_state='NAS task restores usable path I/O. Reverify all192 bound sources,12 prefix files,native imports and resource/dispatch gates before starting the prepared fixed completion.',
    storage_settings_modified=False, existing_data_migration_performed=False,
    scientific_payloads_deleted=False, large_payloads_copied_to_local_or_compute_disk=False)
target = HERE/'DEPLOYMENT_BLOCKER.json'
target.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
rel = HERE.relative_to(ROOT).as_posix()
consumed = PREV/'WAKE1402_AUTOMATION_CONSUMED.json'
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    path = ROOT/name
    d = json.loads(path.read_bytes())
    assert d['active_scientific_jobs'] == 0 and d['pending_job_id'] == receipt['prepared_job_id']
    d.update(status='JOINT_RESPONSE_01_waiting_external_NAS_access', updated_UTC=now,
        execution_state_asof_UTC=receipt_reads[-1]['UTC'], active_scientific_jobs=0,
        current_action='318 saved columns audited; fixed completion fully prepared locally but not dispatched because NAS deployment verification is absent.',
        next_action='After NAS access is restored, verify the thin deployment and all gates, then start only the registered424-column completion.',
        active_scientific_jobs_status_basis='Original scientific and deployment helpers absent; fixed completion was never dispatched.',
        current_stage='4_recentered_derivative_waiting_NAS', active_work_package='4_recentered_derivative_waiting_NAS',
        current_followup_status='NONE_EXTERNAL_NAS_ACTION_REQUIRED', followup_status='NONE_NO_RUNNING_WAIT',
        followup_automation_id=None, followup_automation_status='DELETED_CONFIRMED',
        followup_next_run_local=None, next_wake_local=None, followup_scheduled_local=None,
        followup_receipt_sha256=hashlib.sha256(consumed.read_bytes()).hexdigest(),
        scientific_acceptance=False)
    d['followup'] = dict(status='CONSUMED_DELETED_EXTERNAL_NAS_ACTION_REQUIRED', automation_id='n01', count=0,
        next_wake_local=None, receipt=consumed.relative_to(ROOT).as_posix())
    d['joint_response_01'].update(recentered_derivative_completion_status='registered_not_started_NAS_deployment_blocked',
        recentered_derivative_completion_local_review=rel+'/LOCAL_PREPARATION_REVIEW.json',
        recentered_derivative_completion_storage_blocker=rel+'/DEPLOYMENT_BLOCKER.json',
        recentered_derivative_completion_complete_terminal_audit_executed=False,
        recentered_derivative_combined_complete=False)
    if name == 'jobs.json':
        job = d['jobs'][-1]
        assert len(d['jobs']) == 142 and job['job_id'] == receipt['prepared_job_id']
        job.update(status='registered_not_started_NAS_deployment_blocked', dispatched=False,
            scientific_calls=0, local_preparation_review=rel+'/LOCAL_PREPARATION_REVIEW.json',
            storage_blocker=rel+'/DEPLOYMENT_BLOCKER.json')
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
readme = ROOT/'README.md'
old = readme.read_text(encoding='utf-8')
header = f'''# N01 当前阶段：318列已审核，固定补全等待NAS

> 当前执行状态（{local:%Y-%m-%d %H:%M}，UTC+8）：0科学作业、0定时。原导数采集因NAS写入中断；固定补全已完成本地准备，尚未派发。阶段4科学验收未过，阶段5未进入。

[前缀复核](receipts/joint_response_01_20261020/recentered_derivative_diagnostic_01/terminal_01/PREFIX_REVIEW.json)核验185源码、20输出、6库318列、132有限数组、30720角色身份、36风险视图与6控制门。原程序报告327列，其中9列未落盘、415列未执行；原退出回执缺失，退出码未知。原失败与预算保留。

[固定补全](docs/JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_COMPLETION_20261020.md)只新增8库424列、864生成调用，复用原6库318列，零求解、候选和参数提交。累计751列、1532生成，9重复列和20重复生成费用明确保留。科学设置、模型、损失、硬读出、角色、随机库及科学门不改。

[本地准备审核](receipts/joint_response_01_20261020/recentered_derivative_completion_01/LOCAL_PREPARATION_REVIEW.json)已核对192源文件/191pins、177份Python AST、精确新增及累计预算和独立终止审计入口。监控写入失败后终止拥有子进程的故障测试及NAS原位NPZ刷新替换测试通过。这些是工程证据，完整导数采集和恢复求解仍未完成。

[NAS部署阻塞记录](receipts/joint_response_01_20261020/recentered_derivative_completion_01/DEPLOYMENT_BLOCKER.json)保留两次部署等待超时及轻量部署会话异常。轻量布局只写7份文本、只读复用185份锁定源文件；14:50原位读取确认部署通过回执未找到。全部相关进程已退出，固定补全未启动。没有运行中的等待作业，未安排无依据的轮询；NAS恢复后先完成部署与资源门核验，再启动固定补全。NAS迁移及挂载配置未改动。

所有新科学输出、缓存与临时数组直接留NAS，私有波形、逐行身份、NPZ与权重不上传。以下保留历史快照，当前以本节、三台账和回执为准。

'''
readme.write_text(header+'---\n'+old.split('---\n', 1)[1], encoding='utf-8', newline='\n')
plan = ROOT/'JOINT_RESPONSE_01_PLAN.md'
lines = plan.read_text(encoding='utf-8').splitlines()
lines[2] = f'> 当前执行状态（{local:%Y-%m-%d %H:%M}，UTC+8）：318列保存前缀已审核；固定补全424列已准备但NAS部署门未过，尚未启动。0科学作业、0定时；等待NAS访问恢复，阶段4科学验收未过，阶段5未进入。'
text = '\n'.join(lines)+'\n'
text += '''

## 当前外部前提：NAS访问恢复

固定补全已完成192源文件/191pins、177份Python AST、预算与终止审计入口的本地复核。两个部署进程等待NAS响应超时，轻量部署会话未返回通过；原位查阅确认部署通过回执未找到。所有相关进程已退出，科学补全从未派发。见[阻塞回执](receipts/joint_response_01_20261020/recentered_derivative_completion_01/DEPLOYMENT_BLOCKER.json)。没有运行中的计算或部署，未设置固定轮询；NAS任务恢复可用访问后，先验证全部绑定文件、原位导入及资源，再启动同一已登记固定补全。原job141失败、318列已审核/9列未落盘/415列未执行及所有科学门保留。恢复求解前提尚未满足。
'''
plan.write_text(text, encoding='utf-8', newline='\n')
print(json.dumps(dict(status=receipt['status'], active_jobs=0, followup_count=0, completion_dispatched=False,
    local_preparation_review=review['status'], storage_blocker_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
