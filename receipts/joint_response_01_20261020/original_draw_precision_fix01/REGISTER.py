from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json,shutil

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
previous=root/'receipts/joint_response_01_20261020/original_draw_precision_01'
review=json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
assert review['status']=='PASS_STOPPED_ORIGINAL_PRECISION_PREFIX_AUDIT' and review['committed_parameter_updates']==0
ledger=json.loads((root/'jobs.json').read_bytes());old_jobs=copy.deepcopy(ledger['jobs'])
assert len(old_jobs)==131 and ledger['active_scientific_jobs']==0
(here/'before_registration').mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/name,here/'before_registration'/name)
now=datetime.now(timezone.utc).isoformat()
contract='docs/JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_FIX01_20261020.md'
(root/contract).write_text('''# JOINT_RESPONSE_01：原生成器32draw距离计算修复FIX01

2026-10-20前瞻登记。前项原生成器嵌套4→32draw诊断已保存9列，实测546.638秒，折合60.7376秒/列；108列单独预计6559.657秒，超过原5400秒内部预算，尚未计第二头设置和候选。已核对PID/boot/ticks后SIGTERM停止，原生exit -15，保存9列、5组中心、256行M0零增量；805源/25输入/9输出审计通过，零模型更新。末次快照之后0—9列可能完成而未落盘，费用区间保留。该停止属于运行可行性处理，未形成科学失败判定。

瓶颈对应原native的均匀不等样本W1路径：32768生成行对4096REAL行，逐16列执行CDF合并及搜索。对均匀8:1样本，精确一维W1可把排序后的每8个生成值与一个REAL次序统计量配对，再取绝对差均值。这与原CDF积分定义相同。计算在距离函数内部完成，输入行、源权重、REAL统计分母和Q分箱支持计数保持原值；没有在数据层复制REAL事件。

新12行算子只处理无显式权重且生成行数恰为REAL八倍的情况，其余情况原函数原样执行。局部进程显式绑定run_response036的W1符号及losses008_v2的W1符号，覆盖native波形、继承的QPSD/noise；原源码文件保持字节不变。工程检查包含六组数值及独立SciPy参照、整数比/不等比/等样本数、一维、重复值和可学习非均匀权重；最大差4.552e-15，带权值和权重梯度逐值一致。32768/4096/32列CPU实测0.214秒降至0.083秒；该耗时不直接当作GPU加速结论。

完整科学设计继承[原登记](JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_20261020.md)：原JointResponse01、零M0、旧4draw加独立28draw、固定5角色SOURCE/REAL索引和投影、两头108列、完整联合/native求解、最多6候选/4割线修正、原全部局部门及FRESH选择规则、零参数更新均不变。重建的seed0五组32draw中心及首9列必须和前项保存数组比较：联合残差及参数数组逐值相同，native值rtol=atol=1e-12。通过后继续剩余列；差异立即失败。前9列重算费用计入本次预算，旧作业保留为独立终止记录。

本次重新从M0执行一次完整有界诊断，不加载前项拟合端点；前项本就没有提交端点。原25输入与新增脚本、算子、工程证据及前9列文件分别锁定。科学操作上限沿用原700生成调用/22585600行、2094观察调用/67756032行、native76333056行；新增计数仅5组中心和9列保存证据对照，生成次数不增加。native耗时独立记账，便于按实际速度估时。

仍4CPU、GPU4GiB、RSS12GiB、内部5400秒、外部5700秒加20秒、128MiB输出；原作业不重启、不延预算。当前修复单独登记，一次一个N01科学作业。部署、数值审计准备及能立即执行的工作先完成；正确启动并记录实际列速度后估算完成时刻，加余量安排本聊天单次苏醒。两头局部门通过只支持下一项有界校准，原最终REAL改善门与阶段5继续保留。
''',encoding='utf-8',newline='\n')
prior_name='configs/joint_response_01_original_draw_precision_20261020.json'
p=json.loads((root/prior_name).read_bytes())
p.update(schema='n01/joint-response01-original-draw-precision-fix01/v1',job_id='JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_FIX01_20261020',
    registered_UTC=now,method_contract=contract,parent_stopped_job=p['job_id'],stopped_prefix_columns=9,
    native_distance_definition_unchanged=True,native_8_to_1_implementation_changed=True,old_budget_not_extended=True)
p['operation_budget'].update(same_32draw_centers_verified=5,predecessor_FD_columns_verified=9)
inputs={}
p['stopped_prefix_arrays']='original_precision_fix01_inputs/seed0_MULTIBANK_M0_PREFIX.npz'
inputs[p['stopped_prefix_arrays']]=str(previous/'terminal_01/collected/run_01/seed0_MULTIBANK_M0.npz')
for name in ('RUNTIME_STOP_DECISION.json','RUNTIME_STOP_RECEIPT.json','INTEGER_W1_ENGINEERING_REVIEW.json','terminal_01/TERMINAL_REVIEW.json','AUDIT_STOPPED_PREFIX.py'):
    path=previous/name;inputs[path.relative_to(root).as_posix()]=str(path)
script='scripts/diagnose_joint01_original_draw_precision_fix01.py'
for name in (prior_name,script,contract,'src/n01/joint01_integer_w1.py'):
    p['source_pins'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
for name,local in inputs.items():p['source_pins'][name]=hashlib.sha256(Path(local).read_bytes()).hexdigest()
config='configs/joint_response_01_original_draw_precision_fix01_20261020.json'
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8',newline='\n')
manifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes())
sources=len(set(manifest['files'])|set(inputs)|{config,script,contract,'src/n01/joint01_integer_w1.py'})
ast.parse((root/script).read_text());ast.parse((root/'src/n01/joint01_integer_w1.py').read_text())
(here/'STATIC_REVIEW.json').write_text(json.dumps(dict(status='PASS',UTC=now,
    original_generator_and_loss_files_unchanged=True,metric_definition_unchanged=True,integer_ratio_uniform_only=True,
    no_REAL_row_replication=True,conditional_support_counts_unchanged=True,old_weighted_path_retained=True,
    explicit_process_local_W1_bindings=['run_response036','n01.losses008_v2'],prefix_replay_required_before_further_work=True,
    old_prefix_recompute_counted=True,parameter_updates=0,scientific_acceptance=False),indent=2)+'\n',encoding='utf-8',newline='\n')
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,
    config_sha256=digest,host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_original_draw_precision_fix01_20261020',
    resources=p['resources'],operation_budget=p['operation_budget'],parameter_updates=0,automatic_training=False,
    original_job_not_restarted=True,scientific_acceptance=False))
assert ledger['jobs'][:131]==old_jobs
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=ledger if name=='jobs.json' else json.loads((root/name).read_bytes())
    d.update(status='JOINT_RESPONSE_01_original_draw_precision_fix01_registered',active_scientific_jobs=0,pending_job_id=p['job_id'],
        active_job_id=None,active_run_id=None,training_pid=None,supervisor_pid=None,updated_UTC=now,
        current_stage='4_original_generator_draw_precision_fix01',active_work_package='4_original_generator_draw_precision_fix01',
        scientific_acceptance=False,current_action='Exact 8:1 W1 numerical repair registered with frozen scientific design and prefix replay.',
        next_action='Deploy, verify pinned same 32-draw centers/first nine columns, then complete bounded diagnostic.')
    d['joint_response_01'].update(original_draw_precision_fix01_config=config,original_draw_precision_fix01_config_sha256=digest,
        original_draw_precision_fix01_status='registered_not_started',original_draw_precision_fix01_started=False)
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,
    source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,prior_131_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
