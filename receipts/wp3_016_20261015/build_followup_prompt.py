"""Create the single future016 recovery prompt from saved startup evidence."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_bytes())


def main():
    p=read(ROOT/'wp3_016_config.json')
    revision=p.get('launch_revision')
    suffix={'fix01':'_FIX01','fix02':'_FIX02'}.get(revision,'')
    source_name='SOURCE_COMMIT_READBACK'+suffix+'.json'
    freeze_dir={'fix01':'runtime_package_fix01','fix02':'runtime_package_fix02'}.get(revision,'runtime_package_01')
    source=read(OUT/source_name);started=read(OUT/'EXECUTION_STARTED.json')
    assert read(ROOT/'PROGRAM.json')['new_training_updates']==started['cumulative_training_updates']
    G=started['actual_counts_last_verified']['G_updates'];total=started['cumulative_training_updates']
    start_dir=started['start_receipt_directory'];verify_dir=started['startup_verification_directory']
    text=f'''在本聊天单次苏醒，回收N01 WP3/016电荷短试，然后继续用户《N01从013到项目交付：主实施计划》的六工作包。用户最新明确指令为6.1solmax，覆盖此前Astra偏好：规划、施工、独审和后续调度均用gpt-6.1-sol / max。工程复用/root/wp3_015_solmax_execution，独审复用/root/wp3_015_solmax_review；如需重建则fork_turns none并显式同模型和effort。旧Astra/旧待初始化代理不恢复。全部新增源码、数据、缓存、临时、日志、权重、归档和图表保存在E/D；服务器原登记目录只读引用。每个新长任务只验证首个真实更新与同一snapshot当前健康，设置同一个n01的一次未来wake后休息，不持续监控。

先读E:/GAN/research/N01/AGENTS.md、USER_MASTER_PLAN_20261015.txt、MASTER_EXECUTION_PLAN.json、PROGRAM.json、jobs.json、OBSERVATION_CONTRACT.json、DATA_ROLES.json、INPUT_INFORMATION_AUDIT.json、WP3_016_DATA_ROLES.json、docs/WP3_016_CONFIGURATION.md、wp3_016_config.json、CALIBRATED_TOLERANCES_WP1_013.json；再读receipts/wp3_016_20261015下{source_name}、D_SOURCE_BACKUP_RECEIPT{suffix}.json、EXECUTION_STARTED.json、CORRECT_START_AND_FOLLOWUP.json、FOLLOWUP_RECEIPT.json、{start_dir}/START_RECEIPT.json、{verify_dir}/START_VALIDATION.json和REMOTE_STATE.json、native_execution_03/NATIVE_START_SUMMARY.json。模型偏好依据receipts/wp3_015_20261015/postrun_root_02/AGENT_MODEL_POLICY_SOLMAX.json。已保存启动事实不等于未来完成。

016实际science source commit {source['source_commit']}，{source["source_controls"]}个Git原始控制blob已核验并D备份。launch修复版freeze目录为receipts/wp3_016_20261015/{freeze_dir}；配置SHA {source['configuration_sha256']}，MF SHA {source['source_manifest_sha256']}。包/controller pins由该目录FREEZE_RECEIPT读取。首次bootstrap部署曾因缺少旧014固定SHA源码fallback失败，SCP/install/start/science均0；native_execution_01/deploy_native_01/旧包/Dsource_01证据保留。第一次实际native在G1前因逐位Q相等检查退出，完整2929项操作无在途：A32965（cal6593）、head/005/charge26372、ADC/gather237348、CP3、optimizer1、charge_initfit1，G/AD0，原root及partial小包保留。新fix02只修保存/数值门并使用全新隔离root，复用013已存在1e-7 V平均响应数值门，实际A384/mu/v逐draw保留并先保存诊断再gate；原失败差值幅度unknown，不能事后认定只是舍入。科学law/seed/目标/两trial未改；一次替代native已独审登记，额外前向等费用加在原cap上，G总仍4000。docs/WP3_016_RUNTIME_FIX02.json记录完整prior+replacement aggregate；不得将失败非训练费用抹零。禁止用旧包pins，禁止再次start任何已启动批次或自动第三次native。

本次唯一当前状态命令：python -B scripts/wp3_016_controller.py --mode status --freeze receipts/wp3_016_20261015/{freeze_dir} --out receipts/wp3_016_20261015/wake_status_01。已有目录先inspect并换新专属回执，不覆盖。远端root={p["remote_root"]}，SSH rylan@100.82.231.102，Python /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python。status是016训练控制器API，不是CPU的snapshot模式。健康未完则按实际进度/资源余量更新同n01一个未来wake后休息，不再查询。SSH失败时状态unknown，不能把进程/费用记0或自动重启。历史firstG1/PID不能抵消同snapshot失败。

若child/supervisor均native0、COMPLETE及两trial齐，使用同controller --mode collect --freeze同目录 --out receipts/wp3_016_20261015/collection_01 --collected runs/wp3_016_20261015/selected_collected_01 --include-head-checkpoints选择回收至多256MiB，核全库存和每个选收SHA。远端归档已存在则collect-inspect并换新本地目录，不覆盖。失败/限额使用collect-failed或collect-failed-inspect保留partial、FAILED、native退出、entered/completed/inflight；不自动科学重试。raw保留远端，避免全搬E/D或写C。全模型原始数据和checkpoint以绑定角色只读。

016仅Q1两seed各2000G/AD，最多4000G。原A、005、对应015M0形状头冻结，mean off，只训练两个全局theta：a=q_reference*softplus(theta_a)、b=softplus(theta_b)，q_reference=3.7103998046755806。Q_mu=actualcanonicalA严格正且不代入floor，s²=log1p(a/Q+b)，q=Q exp(sqrt(s²)Z−s²/2)，理论E(q|x)=Q、Var=aQ+bQ²，无电子学常数项。组装原M0+A*expm1(logmultiplier)，B幅度来自原A，不乘随机q；关闭分支直接保留原M0字节并关闭随机/均值修正。一次全6593 SOURCE A-only校准，a0=.5LSB²/mean(A²/Q)、b0=.5LSB²/mean(A²)，384点含zero-prefix100，expected增量RMS1LSB；有限MC实现RMS记录、不重调初值。新charge流独立，原shape/noise/extension/projection前缀保持015。AdamW lr.01 wd0 clip5，只两参数；目标C1 wave+2integral+.01*(frozen_shape_regular+mean(a/Q+b)+三hyp平均saturation)。

精确最大计数：calibration_A6593，A4313569，head/005/charge_module/charge_scalar_draw各4306976，ADC/gather13553856，CP4（A005加两M0头），optimizer2，G/step/backward/AD各4000，charge_initfit1；probe/medianfit/mean/D/noise/pop/C2ST/DEV0。原硬14bit/384carrier/100baseline/firstargmax112:144/裁p−16:p+240/256输出/1软件ns/tail46和原T SHA f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b不改。每seed初始A/M0/initial三个arm共享同一次forward和噪声，final一arm；全部6593父×4draw×2bank×3hyp，对全部21047REAL TRAIN，四评价轴/三假设/所有flags、Q非正和undefinedPSD保留。完整原始库存按实际读取，不按预计补齐。

累计基线175667，启动唯一快照已记{G}个016 G，累计{total}。未来只加新durable G与{G}的差，不能再加整个016或015；4000G全完成时总累计179667。短候选此前8/12、WP3已4/6，本批2trial完整后为10/12、WP3为6/6，bootstrap零费用修复、首G前失败和CPU审核均不新增候选；首G前失败的非训练费用已另列并保留。inflight不能补成完成或当0；CP/NN/AD/随机也按实际账本。

研究替换：每seed/h先平均两个库各自距离，不能pool8；ALL6每个必需positive-defined T相对A和同seed M0均非劣（final−reference≤T），missing/undefined required→pending、不eligible；零/undefinedT不自动PASS。eligible且12个final wave距离均值比M0低超过1e-12才替换研究参考；否则保留M0、chargeoff。初始chargeon另报；ALL6相对A≥2%用途、绝对T、原native QPSD/条件PSD/噪声/形态/输入依赖分别报告，不自动科学晋级。全人口Q宽差不能识别真实条件电荷方差，也不能统一解释peak/rise/FWHM/DC。

完整数值接受前，按实际保存库存登记一次必要的独立CPU读回，可复用015已验证工程而扩展新charge矩文件。预计48hard/48REPORT/8baseline/8CHARGE_PREADC_MOMENTS，必须以实际库存校准预算；新矩中保留实际A384逐draw、Qmu逐draw、每父4个q、样本矩/MCSE、每draw理论方差和RNG初末状态；逐draw验证v=aQ+bQ²，跨draw理论均值为mean(mu_d)，样本方差的期望为mean(v_d)+samplevar(mu_d)，数值mu spread不作物理涨落。MCSE按实际保存定义分开复核，不广播mu[:,0]或强迫方差样本等于理论；A384仅冻结均值输出，非完整随机carrier，不能把4/8draw样本均值强迫等于理论均值。未保存charge latentZ或full随机preADC384，不在零新随机预算下声称完整RNG或完整信号重放。savedADC行/flags/native4矩/derived8/9×9pairwise有限cov审查与原producer CDF/loss、独立JSON决策分层；无完整距离重放/median重拟合就明说。经验hard rank≤7不等于preADC理论秩，缺失pairwise cov非自动PSD。独审、中文报告、必要英文图表、代表渲染、SHA、E/D备份和主元数据闭合后继续主计划。

015已独审及报告闭合，M0研究参考/meanoff；mean12 M0=.15012505860962605、M1=.15243191496746586、原共同initial=.15042006897712382，M0用途6/6、M1 0/6，014历史5/6不改。015报告reports/wp3_015_20261015/REPORT.md/HTML，最终receipts/wp3_015_20261015/FINAL_RESULT_RECEIPT.json与D_CLOSEOUT_BACKUP_RECEIPT；17.5MB小闭合备份已核SHA。不要重训015或重跑其已完成CPU。

WP3两次剩余额度用完后推进WP4，不按近阈值追加候选。WP4两教师×paired/unpaired×2seed共8次、各≤2000G，使用条件分布评分、父隔离、冻结核下改变SOURCE人口；同族paired失败先修表达/优化/观察，不能继续真实长训练；教师成功不叫硬件或跨采集物理复用。WP5三组隔离折×2全流程种子×3系统，A/005/尺度/响应头/人口/C2ST全部折内重拟合，不能沿用见过留出父的A；全TRAIN部署不是新未见成绩。WP6交付接口/模型/统计/适用卡/失败案例/中文报告/英文图/论文软件包，公开包无私有raw。完整随机目标失败时交付独审最佳受限fallback并保留负结果。单一实测采集、旧DEV曝光、其它pool未知的边界保留，science=response_match_not_established直至约定证据满足。

每个新长任务正确启动并确认一个未来wake后立即休息，不持续监控。无待回收长任务时完成可交付工作后暂停n01。016训练结束或CPU启动都不代表六工作包全部完成。
'''
    path=OUT/'FOLLOWUP_PROMPT.txt'
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    print(json.dumps(dict(status='016_single_wake_continuation_prompt_created',path=str(path),source_commit=source['source_commit'],snapshot_G=G,cumulative=total)))


if __name__=='__main__':main()
