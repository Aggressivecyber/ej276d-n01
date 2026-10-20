from datetime import datetime, timezone
from pathlib import Path
import copy, hashlib, json, shutil
import numpy as np

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
OUT=HERE/'terminal_01/collected/run_01';R=HERE.parent/'random_precision_diagnostic_01'
review=json.loads((HERE/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
terminal=json.loads((HERE/'terminal_01/REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status']=='PASS_SAVED_STABLE_PROJECTION_DIAGNOSTIC_AUDIT'
assert not review['two_seed_stable_projection_all_local_pass']
assert terminal['native_exit']['exit_code']==0 and not terminal['processes']
now=datetime.now(timezone.utc).isoformat();diagnosis=[]
for seed in range(2):
    path=OUT/f'seed{seed}_STABLE_PROJECTION.npz'
    with np.load(path,allow_pickle=False) as z:a=dict(z)
    rows=review['summaries'][str(seed)]['by_model']
    removed=a['curvature_update_control']-a['stable_projection']
    iid=[]
    for rep in range(4):
        label=f'IID{rep}'
        with np.load(R/f'terminal_01/collected/run_01/seed{seed}_rep{rep}_IID.npz',allow_pickle=False) as z:b=dict(z)
        actual=(a['curvature_update_control_'+label+'_actual_native']-a['stable_projection_'+label+'_actual_native'])/a[label+'_baseline_native']
        predicted=b['native_jacobian']@removed/a[label+'_baseline_native']
        original=b['jacobian']@(a['curvature_update_control']-a['theta'])
        dropped=b['jacobian']@removed
        iid.append(dict(library=label,relative_native_effect_of_restoring_removed_step=actual.reshape(3,3).tolist(),
            stale_J_predicted_native_effect=predicted.reshape(3,3).tolist(),
            fraction_of_original_linear_joint_effect_removed=float(np.linalg.norm(dropped)/np.linalg.norm(original))))
    diagnosis.append(dict(seed=seed,output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        accepted_libraries={name:sum(v['accepted'] for v in values.values()) for name,values in rows.items()},
        failed_libraries={name:[label for label,v in values.items() if not v['accepted']] for name,values in rows.items()},
        failures_all_native_only=all(v['macro_pass'] and v['trust_pass'] and v['joint_ratio']<=.99 for values in rows.values() for v in values.values()),
        paired_IID_native_and_joint_effects=iid))
record=dict(status='SAVED_PAIRED_PROJECTION_DIAGNOSIS_COMPLETE',UTC=now,heads=diagnosis,
    conclusion='Joint-only weak-direction removal loses native-protection directions: unchanged control passes 8/9 libraries per head; projection passes 4/9 and 5/9. Do not adopt this truncation or tune its threshold.',
    new_generator_calls=0,parameter_updates=0,scientific_acceptance=False,
    next_question='The unprojected fixed control misses only one native-anchor check per head. Assess a separate feasibility-restoration step centered on its measured outputs, without reopening same-center curvature corrections. Old derivatives can screen feasibility but cannot validate a step at the new center.')
(HERE/'PAIRED_PROJECTION_DIAGNOSIS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
before=HERE/'before_closed';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(ROOT/name,before/name)
rel=HERE.relative_to(ROOT).as_posix()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((ROOT/name).read_bytes())
    if name=='jobs.json':
        old=copy.deepcopy(d['jobs']);assert len(old)==140
        d['jobs'][-1].update(status='completed_audited_local_gates_failed',native_exit_code=0,ended_UTC=review['native_exit']['ended_utc'],
            terminal_review=rel+'/terminal_01/TERMINAL_REVIEW.json',execution_counts=review['execution_counts'],parameter_updates=0,two_seed_stable_projection_all_local_pass=False)
        assert old[:-1]==d['jobs'][:-1]
    d.update(status='JOINT_RESPONSE_01_stable_projection_diagnostic_closed_negative',active_scientific_jobs=0,active_job_id=None,
        active_run_id=None,pending_job_id=None,training_pid=None,supervisor_pid=None,current_job_native_exit_code=0,
        current_job_ended_UTC=review['native_exit']['ended_utc'],last_verified_UTC=terminal['UTC'],execution_state_asof_UTC=terminal['UTC'],
        updated_UTC=now,current_action='Fixed projection diagnosis audited negative; original unprojected controls retain native-protection directions.',
        next_action='Use saved measured control outputs to assess feasibility restoration at that new center; no projection-threshold search or added same-center curvature term.')
    d['joint_response_01'].update(stable_projection_diagnostic_status='completed_audited_local_gates_failed',
        stable_projection_diagnostic_completed=True,stable_projection_diagnostic_complete_terminal_audit_executed=True,
        stable_projection_diagnostic_terminal_review=rel+'/terminal_01/TERMINAL_REVIEW.json',
        stable_projection_diagnostic_execution_counts=review['execution_counts'],stable_projection_diagnostic_two_head_gate_pass=False)
    assert d['followup']['count']==0
    (ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
paragraph='''
## 2026-10-20 12:49：固定弱方向投影消融失败，原生保护方向不可按joint单独删减

[终止独立审核](receipts/joint_response_01_20261020/stable_projection_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)确认902份源码/105 pins和13份输出共12190450字节，exit0、67.34秒；72次生成、36个接受判定、162个风险视图、两头投影构造及92160个角色条目复核通过。16个中心、初值和原控制候选精确重放，零新求解或参数更新。完整候选波形及RMS提取未另行独立重生成。

仅按joint稳定性保留6/7维，把原控制步平方范数压至15.5%/69.9%，却使两头接受组数由8/9、8/9降为4/9、5/9。所有失败均为原生wave/QPSD保护；联合1%收益、宏块和输出信赖域仍通过。不能据此删掉这些方向，也不搜索投影阈值。原控制的唯一失败分别为seed0 IID2的QPSD初值比1.010505、seed1 FIT0的1.010555，原1.01界不变。

原控制候选的已测输出可作为新中心，评估最小可行性恢复是否可行；旧导数只能作保存矩阵筛查，若有支持，必须在新中心前瞻登记并重新测导数后才能检查候选。不会追加同中心第三个曲率项。阶段4科学门仍未通过，阶段5未进入，当前无运行作业、无等待定时。
'''
path=ROOT/'JOINT_RESPONSE_01_PLAN.md';lines=path.read_text(encoding='utf-8').splitlines()
lines[2]='> 当前执行状态（2026-10-20）：随机精度和固定弱方向投影诊断均已审核失败；当前分析原控制候选新中心的可行性恢复。0科学作业、0定时；原科学门未过，阶段5未进入。'
path.write_text('\n'.join(lines)+'\n'+paragraph,encoding='utf-8')
path=ROOT/'README.md';_,tail=path.read_text(encoding='utf-8').split('\n---\n',1)
head='''# N01 当前阶段：阶段4固定弱方向投影消融闭合

> 当前执行状态（2026-10-20）：固定投影诊断67.34秒正常结束，独立审计通过、两头局部门失败。当前0科学作业、0定时；REAL响应匹配尚未建立，阶段5待验。

[最近独立审核](receipts/joint_response_01_20261020/stable_projection_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)复核902源码/105 pins、13输出、72生成调用、36接受判定和162风险视图。投影使两头接受组数从原控制的8/9、8/9降为4/9、5/9，全部因原生wave/QPSD保护失败。joint弱方向中包含原生保护所需方向，不能按joint稳定性单独删减，不做阈值搜索。

[保存证据分析](receipts/joint_response_01_20261020/stable_projection_diagnostic_01/PAIRED_PROJECTION_DIAGNOSIS.json)定位原控制仅各有一组QPSD初值比超过1.01（1.010505/1.010555）。下一步先检验在原控制已测输出这一新中心能否做最小可行性恢复；旧导数只作筛查，新执行必须重新登记并测新中心导数。同中心三候选曲率序列与成对随机降方差路线保持失败闭合。

原三轮REAL校准的两神经各24/96最终改善门失败，原5% native改善等科学门不变。以下保留历史快照，当前以本节、三台账和对应回执为准。

'''
path.write_text(head+'\n---\n'+tail,encoding='utf-8')
(HERE/'CLOSURE_RECEIPT.json').write_text(json.dumps(dict(status='CLOSED_AUDITED_LOCAL_GATES_NEGATIVE',UTC=now,
    prior_139_jobs_unchanged=True,active_jobs=0,automations=0,scientific_acceptance=False),indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
