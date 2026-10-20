from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import sys
sys.path.insert(0,'E:/GAN/tools/clarabel_0_11_1')
import clarabel

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
R=HERE.parent/'recentered_restoration_solve_01';R.mkdir(exist_ok=False)
now=datetime.now(timezone.utc).isoformat()
review_path=HERE/'terminal_01/TERMINAL_REVIEW.json';review=json.loads(review_path.read_bytes())
terminal=json.loads((HERE/'terminal_01/REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
m=json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
assert review['status']=='PASS_SAVED_COMPLETED_RECENTERED_DERIVATIVE_AUDIT'
assert review['combined_unique_FD_columns']==742 and review['native_exit']['exit_code']==0
labels=['FIT0','FIT1','FIT2','IID0','IID1','IID2','IID3'];inputs={}
for seed in range(2):
    inputs[str(seed)]={}
    for label in labels:
        name=f'run_01/seed{seed}_{label}_RECENTERED.npz'
        if seed==0 and label!='IID3':
            original=m['prior_NAS_root']+'/'+name;entry=m['external_prefix_files'][original]
        else:
            original=terminal['remote_root']+'/'+name;entry=terminal['outputs'][name]
        inputs[str(seed)][label]=dict(path=original.replace('/mnt/MiNas/','Z:/'),**entry)
source_paths=[ROOT/'scripts/solve_joint01_recentered_restoration.py',ROOT/'scripts/audit_joint01_recentered_restoration.py',
    Path(clarabel.__file__),*Path(clarabel.__file__).parent.glob('*.pyd')]
source_pins={}
for path in source_paths:
    raw=path.read_bytes()
    if path.suffix=='.py':ast.parse(raw.decode('utf-8'))
    source_pins[str(path)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
config=dict(schema='n01/fresh-recentered-restoration-solves/v1',job_id='JOINT_RESPONSE_01_RECENTERED_RESTORATION_SOLVES_20261020',
    registered_UTC=now,derivative_audit=dict(path=str(review_path),sha256=hashlib.sha256(review_path.read_bytes()).hexdigest()),
    libraries=labels,inputs=inputs,source_pins=source_pins,radius=.003125,native_linear_cap_factor=1.005,
    original_native_execution_factor=1.01,joint_cap=.99,macro_cap=1.15,macro_sampling_allowance=2/4096,
    objective='minimum Euclidean response correction norm',response_coordinates=53,noise_correction=0,
    derivative_center='Same fixed control center as saved residual/native intercepts; fresh complete J/N.',
    solver=dict(max_iterations=100,time_limit_seconds=30,tolerance_feasibility=1e-10,tolerance_gap=1e-10),
    resources=dict(CPU_threads=4,GPU_GiB=0,RSS_GiB=2,wall_seconds=300,output_MiB=16),
    operation_budget=dict(local_solves=2,matrix_libraries=14,response_coordinates=53,joint_constraints=14,
        macro_constraints=168,native_linear_constraints=84,generator_calls=0,actual_candidate_evaluations=0,
        parameter_updates=0,REAL_CHECK_rows=0,DEV_rows=0),
    NAS_output='Z:/GAN/research/N01/server_runs/joint_response01_recentered_restoration_solve_20261020/run_01',
    report=str(R/'SOLVE_RESULT.json'),progress=str(R/'PROGRESS.json'),audit_report=str(R/'INDEPENDENT_AUDIT.json'),
    stop_rule='Exactly one solve per head. Both independently feasible/KKT-certified before separately registering actual fixed-candidate hard-readout checks. No re-solves, threshold/radius search, curvature changes or automatic training.',
    storage_changes=False,large_local_or_compute_disk_backups=False,scientific_acceptance=False)
cp=ROOT/'configs/joint_response_01_recentered_restoration_solves_20261020.json'
assert not cp.exists()
cp.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8',newline='\n')
digest=hashlib.sha256(cp.read_bytes()).hexdigest()
rel=HERE.relative_to(ROOT).as_posix();rrel=R.relative_to(ROOT).as_posix()
doc='''# 新中心固定恢复求解\n\n742个新中心完整导数及保存中心/角色/控制门已独立审核通过；原job141的失败与重复成本保留。两头各使用FIT0–2和IID0–3全部七库，以新中心残差及原生距离为截距、新中心完整J/N为导数；旧中心风险和原始初值继续作为约束参照。\n\n每头恰好一次Clarabel求解，最小化修正欧式范数。53个响应坐标、noise固定零、半径0.003125；每库联合平方残差≤旧中心0.99倍，各宏块≤1.15倍旧值加2/4096，wave/QPSD线性预测≤旧值和初值中较小者的1.005倍；参数边界保留，无曲率补项。每次最多100迭代/30秒，feasibility/gap为1e-10；CPU4、GPU0、RSS2GiB、总300秒、输出16MiB，矩阵证据直接写NAS。生成器、实际候选、参数更新、CHECK与DEV全部零。\n\n独立审核重建所有10709×53约束矩阵；可行解需原/对偶锥、驻点与互补残差全部达到1e-8，并直接复算约束和参数边界。不可行时核验对偶证书；数值未定时保留未定。只有两头均可行且KKT通过，才另行登记一次固定候选实际硬读出；任何其他结果均不追加求解、不改变半径或余量。局部可行性不代表阶段4科学成功，阶段5仍未进入。\n'''
(ROOT/'docs/JOINT_RESPONSE_01_RECENTERED_RESTORATION_SOLVES_20261020.md').write_text(doc,encoding='utf-8',newline='\n')
for filename in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    path=ROOT/filename;d=json.loads(path.read_bytes())
    assert d['active_job_id']=='JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_COMPLETION_20261020'
    d.update(status='JOINT_RESPONSE_01_fresh_restoration_registered',updated_UTC=now,
        execution_state_asof_UTC=terminal['collection_completed_UTC'],active_scientific_jobs=0,active_job_id=None,
        pending_registered_job_id=config['job_id'],current_stage='4_fresh_derivative_restoration',
        active_work_package='4_fresh_derivative_restoration',
        current_action='Completed job142 audited:424 new plus318 reused equals742 unique FD columns,192 sources/25 outputs and counts/resources verified. Registered one fresh-center restoration solve per head.',
        next_action='Execute the fixed two local restoration solves and independently rebuild matrices/check KKT; register actual fixed-candidate readout only if both pass.',
        active_scientific_jobs_status_basis='Completion native exit0 and no remaining run processes independently verified; no current scientific process.',
        scientific_acceptance=False)
    d['joint_response_01'].update(recentered_derivative_completion_status='completed_audited',
        recentered_derivative_completion_verified_UTC=now,recentered_derivative_completion_native_exit_code=0,
        recentered_derivative_completion_terminal_review=rel+'/terminal_01/TERMINAL_REVIEW.json',
        recentered_derivative_completion_complete_terminal_audit_executed=True,recentered_derivative_combined_complete=True,
        recentered_derivative_combined_unique_columns=742,recentered_derivative_completion_new_FD_columns_last_verified=424,
        recentered_derivative_completion_execution_counts_last_verified=review['execution_counts'],
        recentered_restoration_status='registered_not_started',recentered_restoration_config=cp.relative_to(ROOT).as_posix(),
        recentered_restoration_config_sha256=digest,current_scientific_acceptance=False)
    if filename=='jobs.json':
        assert len(d['jobs'])==142
        prior=hashlib.sha256(json.dumps(d['jobs'][:141],sort_keys=True).encode()).hexdigest()
        d['jobs'][-1].update(status='completed_audited',verified_UTC=now,native_exit_code=0,
            terminal_review=rel+'/terminal_01/TERMINAL_REVIEW.json',complete_terminal_audit_executed=True,
            execution_counts=review['execution_counts'],execution_counts_last_verified=review['execution_counts'],
            combined_unique_FD_columns=742,scientific_acceptance=False)
        d['jobs'].append(dict(job_id=config['job_id'],status='registered_not_started',registered_UTC=now,
            config=cp.relative_to(ROOT).as_posix(),config_sha256=digest,operation_budget=config['operation_budget'],
            resources=config['resources'],source_files=len(source_pins),matrix_inputs=14,
            automatic_restart=False,automatic_training=False,scientific_acceptance=False))
        assert hashlib.sha256(json.dumps(d['jobs'][:141],sort_keys=True).encode()).hexdigest()==prior
    path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
receipt=dict(status='PASS_COMPLETION_CLOSED_AND_FIXED_RESTORATION_REGISTERED',UTC=now,config_sha256=digest,
    job_count=143,prior_141_jobs_unchanged=True,job142='completed_audited',job143='registered_not_started',
    active_scientific_jobs=0,followup_count=0,scientific_acceptance=False)
(R/'REGISTRATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt))
