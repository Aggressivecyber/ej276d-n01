from datetime import datetime,timezone
from pathlib import Path
import ast,hashlib,json

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
now=datetime.now(timezone.utc).isoformat()
old=json.loads((ROOT/'configs/joint_response_01_recentered_derivative_completion_thin_20261020.json').read_bytes())
restoration=HERE.parent/'recentered_restoration_solve_fix01'
certificate=json.loads((restoration/'INDEPENDENT_AUDIT.json').read_bytes())
assert certificate['two_seed_local_feasible'] and certificate['status']=='PASS_INDEPENDENT_FRESH_RESTORATION_MATRIX_AND_CERTIFICATE_AUDIT'
solves=json.loads((restoration/'SOLVE_RESULT.json').read_bytes())
assert solves['execution_counts']['local_solves']==2 and solves['execution_counts']['generator_calls']==0
old_manifest=json.loads((HERE.parent/'recentered_derivative_diagnostic_01/SOURCE_MANIFEST.json').read_bytes())
completion_manifest=json.loads((HERE.parent/'recentered_derivative_completion_01/SOURCE_MANIFEST.json').read_bytes())
prior=completion_manifest['prior_NAS_root']
remote='/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_candidate_check_20261020'
windows=remote.replace('/mnt/MiNas/','Z:/')
doc='''# 固定恢复候选实际硬读出检查

本项仅执行已通过独立KKT审核的两个固定恢复候选。每头FIT0–2、IID0–3共七库，复用原1024父×32draw及4096 REAL拟合行、原随机种子和noise、原三种假设及投影；全部库均已用于约束拟合，结果只属于已暴露库上的诊断。噪声坐标精确保留原中心，仅响应53维应用已保存修正。无新增求解、回溯、阈值/半径搜索或参数提交。

每库先重放旧ITER02中心并核对保存残差/native，再重放固定控制中心并核对保存结果，最后执行唯一固定候选。原门限保留：联合风险≤0.99旧值；四宏块逐假设≤1.15旧值+2/4096；wave/QPSD≤旧值与M0初值的1.01倍，noise≤1.10；相对RMS均值≤0.08、p99≤0.25，flags计数增加≤32768×0.005。原分母及全部行保留，1.005仅是已完成线性求解的设计余量。

预算：14随机库、14 noise005/458752行、1376256扩展行、3670016高斯值；42次生成/1376256行，126次硬观察/4128768行，native4644864行；14旧中心、14控制、14候选检查、14初值anchor复用，REAL拟合57344行。6检查点、TRAIN/PCA各一，A0 52调用6593行，M0 104调用13186行，REAL观察1/21047行，joint/native映射各3。CPU4、GPU4GiB、RSS12GiB、内部600秒、外部900秒并20秒kill余量、输出16MiB；缓存及输出直接使用现有NAS路径。

完成需退出0、全部哈希/计数/资源及保存残差/native/风险/守门独立复核。原始完整波形和RMS提取不在保存数组审计中重新生成；原生读出代码复用固定源码。14个固定候选门全部通过才说明该固定步在这些库上有效；任一失败则闭合失败，不增加候选或放宽门限。无论诊断结果如何，阶段4科学验收仍需原独立重复与改善标准，阶段5未进入。不存在本项自动训练、CHECK/DEV/AD或分类器工作。
'''
doc_path=ROOT/'docs/JOINT_RESPONSE_01_RECENTERED_CANDIDATE_CHECK_20261020.md'
assert not doc_path.exists();doc_path.write_text(doc,encoding='utf-8',newline='\n')
small={
    'scripts/check_joint01_recentered_restoration.py':ROOT/'scripts/check_joint01_recentered_restoration.py',
    'scripts/supervise_joint01_nas.py':ROOT/'scripts/supervise_joint01_nas.py',
    'docs/JOINT_RESPONSE_01_RECENTERED_CANDIDATE_CHECK_20261020.md':doc_path,
    'diagnostic_inputs/RESTORATION_INDEPENDENT_AUDIT.json':restoration/'INDEPENDENT_AUDIT.json',
    'diagnostic_inputs/DERIVATIVE_TERMINAL_REVIEW.json':HERE.parent/'recentered_derivative_completion_01/terminal_01/TERMINAL_REVIEW.json'}
sources={prior+'/'+name:entry for name,entry in old_manifest['files'].items()}
assert len(sources)==185
for name,path in small.items():
    raw=path.read_bytes()
    if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
    sources[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
candidates=[]
for record in solves['reports']:
    name=record['payload'].replace('\\','/').replace('Z:/','/mnt/MiNas/')
    candidates.append(name);sources[name]=dict(bytes=record['payload_bytes'],sha256=record['payload_sha256'])
solve_cp=json.loads((ROOT/'configs/joint_response_01_recentered_restoration_solves_fix01_20261020.json').read_bytes())
p=dict(schema='n01/fixed-restoration-candidate-hard-readout/v1',job_id='JOINT_RESPONSE_01_RECENTERED_CANDIDATE_CHECK_20261020',
    registered_UTC=now,method_contract=doc_path.relative_to(ROOT).as_posix(),
    **{k:old[k] for k in ('frozen_setup_config','roles_file','batch','solver','native_guard','success','saved_arrays','feature_seeds','native_projection_seeds')},
    candidate_arrays=candidates,restoration_evidence_file='diagnostic_inputs/RESTORATION_INDEPENDENT_AUDIT.json',
    derivative_inputs=solve_cp['inputs'],source_pins={name:entry['sha256'] for name,entry in sources.items()},
    resources=dict(CPU_threads=4,GPU_GiB=4,RSS_GiB=12,internal_seconds=600,hard_wall_seconds=900,output_MiB=16,GPU_free_required_MiB=6144),
    operation_budget=dict(checkpoint_loads=6,TRAIN_archives_decoded=1,PCA_archives_decoded=1,
        old_A_calls=52,old_A_rows=6593,M0_head_calls=104,M0_head_rows=13186,
        REAL_observable_replay_calls=1,REAL_observable_replay_rows=21047,feature_maps_constructed=3,new_native_projection_contexts=3,
        allocated_banks=14,shape_standard_normal_values=3670016,noise005_calls=14,noise005_rows=458752,extension_rows=1376256,
        REAL_fit_feature_rows=57344,REAL_CHECK_feature_rows=0,joint_response_calls=42,joint_response_rows=1376256,
        generated_observation_calls=126,generated_observation_rows=4128768,native_component_rows=4644864,
        saved_previous_center_replays_verified=14,saved_initial_anchors_reused=14,saved_new_center_replays_verified=14,
        fixed_control_checks=14,fixed_parameter_candidates=2,fixed_candidate_library_checks=14,
        finite_difference_columns=0,joint_direction_solves=0,REAL_response_updates=0,DEV_waveforms_decoded=0,AD_queries=0,classifier_fits=0),
    noise_correction=0,storage=dict(remote_root=remote,windows_root=windows,large_local_or_compute_disk_backups=False),
    stop_rule='Complete all14 fixed library checks; no extra solve/candidate/radius/threshold change. Retain any failure. No model update or scientific acceptance from exposed-fit diagnostics.')
cp=ROOT/'configs/joint_response_01_recentered_candidate_check_20261020.json'
assert not cp.exists();cp.write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
raw=cp.read_bytes();digest=hashlib.sha256(raw).hexdigest();name=cp.relative_to(ROOT).as_posix()
small[name]=cp;sources[name]=dict(bytes=len(raw),sha256=digest)
manifest=dict(schema='n01/fixed-candidate-source/v1',config_sha256=digest,prior_NAS_root=prior,
    files=sources,external_immutable_sources=185,local_small_files={name:str(path) for name,path in small.items()},
    candidate_payloads=2,bound_files=len(sources))
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
rel=HERE.relative_to(ROOT).as_posix()
for filename in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    path=ROOT/filename;d=json.loads(path.read_bytes())
    assert d['active_scientific_jobs']==0 and d['active_job_id'] is None
    d.update(status='JOINT_RESPONSE_01_fixed_candidate_check_registered',updated_UTC=now,pending_registered_job_id=p['job_id'],
        current_action='Registered the two KKT-certified fixed candidates for original hard-readout checks on14 exposed libraries; source/model/roles/random streams/gates frozen.',
        next_action='Verify existing scientific paths, source bindings and resources; dispatch the registered fixed check if available. No NAS administration.')
    d['joint_response_01'].update(recentered_candidate_check_status='registered_not_started',
        recentered_candidate_check_config=cp.relative_to(ROOT).as_posix(),recentered_candidate_check_config_sha256=digest,
        recentered_candidate_check_remote_root=remote,current_scientific_acceptance=False)
    if filename=='jobs.json':
        assert len(d['jobs'])==144
        d['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,
            config=cp.relative_to(ROOT).as_posix(),config_sha256=digest,operation_budget=p['operation_budget'],resources=p['resources'],
            source_files=len(sources),source_pins=len(p['source_pins']),remote_root=remote,
            automatic_restart=False,automatic_training=False,scientific_acceptance=False))
    path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
receipt=dict(status='PASS_FIXED_CANDIDATE_HARD_READOUT_CHECK_REGISTERED',UTC=now,config_sha256=digest,
    source_files=len(sources),source_pins=len(p['source_pins']),fixed_candidates=2,library_checks=14,generator_calls_budget=42,
    active_scientific_jobs=0,followup_count=0,scientific_acceptance=False)
(HERE/'REGISTRATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt))
