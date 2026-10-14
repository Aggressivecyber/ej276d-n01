"""Register a bounded new head trial without changing immutable endpoints."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = 'shape010_20261014'
REMOTE = '/home/rylan/projects/ej276d-n01/' + RUN
OLD = '/home/rylan/projects/ej276d-n01/window008_resume01_20261013'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value, *, new=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if new else 'w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    stamp = datetime.now(timezone.utc).isoformat()
    before = ROOT / 'receipts' / RUN / 'registration_before'
    before.mkdir(parents=True, exist_ok=False)
    for path in (ROOT / 'PROGRAM.json', ROOT / 'jobs.json', ROOT.parent.parent / 'CURRENT_SIM2REAL_PROGRAM.json'):
        shutil.copy2(path, before / path.name)
    policy = dict(schema='n01/shape010-policy/v1', run_id=RUN, registered_UTC=stamp,
        user_authorization='2026-10-14: 设置计划并推进', remote_root=REMOTE,
        immutable_endpoint_root=OLD, arm='frozen_A_voltage_shape_noGAN',
        protocol='docs/SHAPE010_PROTOCOL.md', TRAIN_counts=dict(source=6593, real=21047),
        G_updates=2000, D_updates=0, noise_updates=0, population_updates=0, C2ST_updates=0,
        hypotheses=['fast','slow','common_level'], hypothesis_reduction='mean_of_separate_losses',
        shape=dict(class_name='n01.model010.FrozenDeterministicVoltageShape010', basis_count=8,
            head_kernel=5, control_count=48, head_initialization_std=.01, voltage_gain=.05,
            mean_model='n01.model008.N01Response008:deterministic:19000',
            deterministic_amplitude='peak_absolute_voltage_of_frozen_A_signal',
            zero_canonical_Q=True, zero_signal_prefix_samples=100,
            charge_randomness=False, time_randomness=False, local_signal_noise=False),
        training=dict(source_parents_per_update=512, draws_per_parent=2, real_rows_per_update=1024,
            forward_chunk_rows=256, sampler='FullCoverageSampler_all_legal_TRAIN',
            G_lr=.0002, G_weight_decay=.0001, clip_norm=5.,
            distribution_loss_stage='joint_stochastic', wave_weight=1., QPSD_weight=2.,
            regular_weight=.01, adversarial_weight=0., discriminator_created=False,
            regularization='actual_voltage_RMS_squared_and_second_difference_RMS_squared_over_a_ref_squared_plus_mean_ADC_saturation',
            checkpoint_interval=1000, status_interval=100, early_stop_by_metric=False),
        seeds=dict(initialization=2026101420, source_sampler=2026101421, real_sampler=2026101422,
            train_shape=2026101423, train_noise005=2026101424, train_extension=2026101425,
            train_projection=2026101426, evaluation_shape=2026101427,
            evaluation_noise005=2026101428, evaluation_extension=2026101429,
            evaluation_projection=2026101430, real_selection=2026101431, source_selection=2026101432),
        evaluation=dict(split='TRAIN', source_parents=512, draws_per_parent=4, matched_rows=2048,
            source_selection='uniform_without_replacement_from_all_legal_SOURCE_TRAIN',
            real_reference_rows=2048, real_control_rows=2048, REAL_REAL_disjoint=True,
            arms=['frozen_A','shape010_initial','shape010_final'],
            same_source_real_latent_projection_bank=True, chunk_rows=128,
            before_training_fixed_baselines=True, final_evaluation_after_update=2000,
            preserve_parent_clusters=True, scientific_pass_threshold=None,
            uses_DEV=False, adaptive_settings=False),
        observation=dict(internal_samples=384, baseline_samples=100, effective_samples=256,
            peak_search=[112,144], samples_before_peak=16, ADC_once=True, canonical_Q_slice=[110,366]),
        resource_limits=dict(GPU_GiB=10, job_tree_RSS_GiB=8, wall_seconds=7200,
            output_MiB=1536, poll_seconds=10, TERM_grace_seconds=20), CPU_threads=12,
        preflight=dict(minimum_free_disk_GiB=4, minimum_available_RAM_GiB=10),
        one_active_scientific_job=True, DEV_numeric_read=False, overwrite_outputs=False,
        automatic_restart=False, scientific_status='response_match_not_established',
        independent_acquisition_available=False, candidate_selection_source='completed frozen TRAIN mean_random009 diagnosis')
    write(ROOT / 'shape010_policy.json', policy, new=True)
    protocol = '''# N01 010：冻结确定性均值的电压域随机形状试验

用户要求“设置计划并推进”。009 已以原生退出 0 完成 512 父 × 16 次的冻结 TRAIN 配对诊断；本轮按已登记的条件计划推进一个预算为 2000 G 的候选。

## 诊断依据

009 中全波形距离 A 约 0.186–0.187，B 约 0.530–0.532，C 约 0.399–0.402；QPSD 则 B 约 0.057–0.060、A 约 0.081–0.091、C 约 0.285–0.312，不能用单项改善代表响应匹配。三种假设结论一致。

逐父纯信号差异 RMS 的均值：D−C 0.003166 V，对应 MCSE RMS 0.003902 V；D−A 0.019859 V；C−A 0.019545 V。16 次重复尚未分辨 D−C 的条件均值漂移。C 与 A 的差异持续存在于 D，支持单独固定 A 的均值进行受控试验，不能据此认定旧均值路径是唯一原因。

## 模型和优化

冻结原 deterministic 19k A 的 shape_encoder、charge_head、FIR、mean_time、U-Net、mean_head、全部缓冲及 stage；只学习一个新的 8 通道形状头。头以冻结 U-Net 的 decoded 特征作为输入，输出 48 个平滑控制点，经 tanh、既有 cubic expansion 和 gate 后投影为 canonical 零电荷基。

信号为 A(x) + Σ B_k(x) z_k，其中 z_k 是独立标准正态，B 的幅度仅依赖冻结 A 的确定性峰幅：0.05 × peak_abs(A)。不进行形状 Q 除法或单位 RMS 归一化。头权重 std=0.01、偏置 0，初始化小且非零。每个 B 的前 100 点严格为零，canonical [110,366) 电荷仅有浮点误差。零 latent 输出与 A 逐字节相同。该构造保持 preADC 条件均值；ADC、寻峰、裁窗后的 Q/PSD 仍须实测。

参数白名单仅新头权重和偏置。随机前向、参数冻结、分布损失模式与 D 更新分别控制；不调用旧 set_stage 解除冻结。只运行 noGAN 一个臂，G=2000，D/噪声/人口/C2ST=0。预算结束即结束，不依据评价提前停止、改设置或扩展训练。

全体合法 TRAIN：SOURCE 6593、REAL 21047。使用 FullCoverageSampler 循环，保留身份和原物理权重；模型仅接收波形。每步 512 源父 × 2 次响应，共 1024 fake，与 1024 REAL 做不配对分布比较，前向分块 256。共享信号在 fast、slow、common_level 下观察，原 005 噪声及延拓模型冻结。

损失为 mean_h(wave + 2 QPSD) + 0.01 × regular。wave/QPSD 使用原 N01Loss008 的 empirical distribution 模式及逐 G 更新的显式投影。regular 包含实际电压形状能量、二阶差分能量（均除以固定 a_ref²）以及平均 ADC 越界惩罚。AdamW lr=0.0002、weight_decay=0.0001、clip_norm=5。原均值模型的常数正则和原 baseline-noise 损失不计入可训练目标。

## 固定回收比较

训练前保存 initial checkpoint，并使用预先固定的 512 SOURCE TRAIN 父（均匀无放回）×4次、2048 REAL 参考和另外 2048 个互不重叠 REAL 对照。用相同源、REAL、随机形状、005、延拓和评价投影比较冻结 A、初始 010、2000 G 终点；基准不用于选择超参数。保存每假设的硬观测数组、距离分解及工程量。生成重复的独立源单位为 512 父，2048 行不是 2048 个独立源父。该 TRAIN 比较是候选诊断，不是留出科学验收。

检查 zero latent 精确 A、零 canonical Q、prefix100、实际梯度、冻结参数和缓冲 hash、优化器白名单、显式随机流、全 TRAIN 覆盖、尝试/成功更新计数及原生退出。第一个真实更新完成后记录启动证明。初始、1000、2000 checkpoint 保存优化器、采样器和随机状态；失败保留最后完整边界及不完整尝试，不自动重启。

## 作业与下一步

服务器新目录 /home/rylan/projects/ej276d-n01/shape010_20261014；008/009 原件保留，只读引用。启动前核对无其它科学作业并冻结源码/输入 SHA256。GPU 10 GiB、进程树 RSS 8 GiB、CPU 12、墙钟 7200 秒、输出 1536 MiB。仅监督本作业进程树。

确认正确启动后，设置一次预计完成时的定时续接，不持续轮询。续接回收原生退出和文件，核验后报告 frozen A / initial / final 的共同随机比较，再决定是否登记同初始化和预算的 GAN 对照或梯度诊断。科学状态保持 response_match_not_established，无自动扩展授权。
'''
    (ROOT / 'docs/SHAPE010_PROTOCOL.md').write_text(protocol, encoding='utf8')
    for old, new in (('start_mean_random009_remote.py','start_shape010_remote.py'),
                     ('supervise_mean_random009.py','supervise_shape010.py'),
                     ('snapshot_mean_random009.py','snapshot_shape010.py')):
        text = (ROOT / 'scripts' / old).read_text(encoding='utf8')
        text = text.replace('mean_random009', 'shape010').replace('MEAN_RANDOM009', 'SHAPE010')
        text = text.replace('n01.diagnose_shape010', 'n01.train_shape010')
        text = text.replace('frozen diagnostic', 'bounded shape-head trial')
        text = text.replace("optimizer_updates=0)", "registered_G_budget=policy['G_updates'])") if new == 'supervise_shape010.py' else text
        if new == 'start_shape010_remote.py':
            text = text.replace("'scripts/supervise_shape010.py',", "'scripts/supervise_shape010.py', 'scripts/supervise_mean_random009.py',")
        if new == 'snapshot_shape010.py':
            text = text.replace('START_PROOF.json', 'FIRST_UPDATE_PROOF.json').replace('SUMMARY.json', 'COMPLETE.json')
        (ROOT / 'scripts' / new).write_text(text, encoding='utf8')
    program = json.loads((ROOT / 'PROGRAM.json').read_text(encoding='utf8'))
    program.update(status='shape010_registered_engineering_before_frozen_launch',
        current_stage='shape010_model_and_driver_engineering', latest_run_id=RUN,
        active_scientific_jobs=0, training_pid=None, current_arm=None, updated_UTC=stamp,
        shape010_configuration='shape010_policy.json', shape010_protocol='docs/SHAPE010_PROTOCOL.md',
        shape010_G_budget=2000, shape010_G_updates=0, shape010_D_updates=0,
        mean_random009_interpretation='full stochastic has poorer waveform; C/A mean paths separated; 16 repeats do not resolve preADC D-C mean drift',
        next_action='Complete synthetic engineering and independent review; freeze inputs and dispatch the bounded noGAN shape010 trial.')
    write(ROOT / 'PROGRAM.json', program)
    jobs = json.loads((ROOT / 'jobs.json').read_text(encoding='utf8'))
    records = jobs['jobs']
    if any(v.get('job_id') == 'N01_SHAPE010_20261014' for v in records):
        raise ValueError('010 already registered')
    records.append(dict(job_id='N01_SHAPE010_20261014', run_id=RUN,
        status='registered_engineering_not_dispatched', registered_UTC=stamp,
        remote_root=REMOTE, policy='shape010_policy.json', protocol='docs/SHAPE010_PROTOCOL.md',
        G_update_budget=2000, actual_G_updates=0, D_updates=0,
        noise_updates=0, population_updates=0, C2ST_updates=0,
        DEV_numeric_read=False, scientific_status='response_match_not_established'))
    jobs['updated_UTC'] = stamp
    write(ROOT / 'jobs.json', jobs)
    current_path = ROOT.parent.parent / 'CURRENT_SIM2REAL_PROGRAM.json'
    current = json.loads(current_path.read_text(encoding='utf8'))
    current.update(status=program['status'], current_stage=program['current_stage'],
        active_scientific_jobs=0, latest_run_id=RUN, updated_UTC=stamp, next_action=program['next_action'])
    write(current_path, current)
    print(json.dumps(dict(status=program['status'], policy_sha256=sha(ROOT / 'shape010_policy.json'),
        G_budget=2000, scientific_updates_added=0, original_endpoints_modified=False)))


if __name__ == '__main__':
    main()
