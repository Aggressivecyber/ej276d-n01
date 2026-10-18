from pathlib import Path
from datetime import datetime,timedelta,timezone
import json,tomllib
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;read=lambda p:json.loads(p.read_text(encoding='utf-8'));path=Path('C:/Users/10629/.codex/automations/n01/automation.toml');before=tomllib.loads(path.read_text(encoding='utf-8'))
(w/'AUTOMATION_BEFORE_UPDATE.toml').write_bytes(path.read_bytes());(w/'AUTOMATION_BEFORE_UPDATE.json').write_text(json.dumps(before,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
a=read(w/'STARTUP_ASSESSMENT.json');d=read(w/'DISPATCH_RECEIPT.json');pkg=read(w/'PACKAGE_RECEIPT.json');deadline=datetime.fromisoformat(d['UTC'])+timedelta(seconds=920)
next_utc=max(datetime.fromisoformat(d['UTC'])+timedelta(minutes=6),datetime.now(timezone.utc)+timedelta(minutes=2));next_utc=next_utc.replace(second=0,microsecond=0)+timedelta(minutes=1);next_local=next_utc.astimezone(timezone(timedelta(hours=8)));next_wake=next_local.isoformat()
(w/'NEXT_WAKE.json').write_text(json.dumps(dict(next_local_wake=next_wake,hard_deadline_including_grace=deadline.isoformat(),schedule_basis='One bounded wake about six minutes after dispatch; original hard budget unchanged.'),indent=2)+'\n')
first=read(w/'STARTUP_RAW.json')['files']['run_01/seed0_SPLITWARP.json'];identities={k:{n:a[k][n] for n in ('pid','start_ticks')} for k in ('dispatcher_identity','supervisor_identity','timeout_identity','native_identity')}
prompt=before['prompt'].split('当前具体交接：')[0]+f'''当前具体交接：本次苏醒完整收集核验、分析、报告并闭合069，随后实现、注册并正确启动070。全部由当前主助手执行，不使用子代理。最新goal与三账本为准，连续推进责任不变。

069闭合CLOSED_NEGATIVE_CROSSING_LEVEL_VOLTAGE_WARP：exit0，原生153.687042秒，峰RSS2808012800B；112文件42560771B/run42467707B，归档SHA167882f4930efedb565f319e69d0c13c8ebec2983c6de828979ae4b30b903223。459pins/468源码、18H精确计数、原06812H完整指标报告、72NPZ、9profiles73791行272cells、156mean/random、128原EX/384native/128carrier、全13186查询/七结点对与108W1通过；实际最终A0/M0/005深冻结通过。证据receipts/response069_20261018/wake_20261018_2019_01。reports/response069_20261018/REPORT.md、SUMMARY.json及DIAGNOSTIC/PEAK_EXAMPLES/TAIL_DIAGNOSTIC三张图已目视核验。未导出完整全人口波形，不能声称完整wave/QPSD离线重算。报告SHA903b1426a634f23d582a32b73eac101c50fbd0ecf40c859415b2e933efa6caab。

069六H wave改善28.275%–30.731%，joint57.942%–58.211%，三形态3.6598–3.6840T，全部点目标过；FWHM/PSD_guarded保护恢复，但全部H仍失败QPSD/tail距离。首头fast FWHM距离M0/068/069=2.468600/3.043903/.765689ns，尾部=.016953/.026042/.039018Vns。固定WARP068输出Q八箱宽度均值[11.270520,14.003458,17.000240,18.420888,20.439368,22.572222,25.238668,27.542446]，REAL[11.082913,14.643790,16.687616,18.755248,20.821428,22.968999,25.308098,28.778906]。宽度总/箱内/箱间方差31.806304/5.419627/26.386677对REAL35.634358/6.214595/29.419764。所有确定性峰保持，FWHM未定义0，rise未定义37–42保留且原保护过，7项observation flags全0；mean→signal392/424，mean→native580–666，relative068 native634–689移峰。

关键新证据见TAIL_AND_CONDITIONAL_PSD_AUDIT.json：原BASE054与069确定性峰和总Q逐行一致，但5350/6593行尾部减少，均值−.030118638Vns；原BASE054/069/REAL尾均值1.016213634/.986094996/1.022874155。原生tail相对BASE054改变−.02821至−.02887，剩余读出差分仅+.00125至+.00191。剩余项合并随机响应/电子学/基线/ADC/选峰窗口，不能当单一噪声因果贡献。BASE054在同两头三H QPSD/tail均在原039上限内，原055失败确认仍关闭。069确定性右半高点全部在原峰p+4.37至p+25.69。全局PSD W1过但条件PSD仍异，QPSD原定义还含逐Q箱PSD分布距离；SOURCE/REAL未配对，Q非能量真值，人口方差非重复输入随机方差。

070唯一候选：docs/RESPONSE070_SPLIT_AREA_CROSSING_LEVEL_WARP_CONTRACT_20261018.md、scripts/run_response070.py/test_response070.py，configSHA{d['config_sha256']}。保留原BASE054均值m、原062八logQ条件SOURCE/REAL profile以及069左右最邻近50/90%交点的七结点/66查询。按原p=argmax m[112:144]，活动[110,176)在p+30分为F/T；30来自原读出tail起点，不拟合边界。mask=(m>0且位置>=p−14)，P=mask*m，W=原电压插值加相同mask。每段A_j=sum P_j，V_j=sum(WP)_j，α_j=A_j/V_j，m'=m+α_j(WP)_j−P_j。两段面积须正，否则整作业显式失败，不丢行/回退。

r=M0−A0，R=mask*r，每段B_j=α_j(WR)_j，u_j=α_j(WP)_j/A_j，random校正=B_j−(sumB_j−sumR_j)u_j−R_j。原每段随机积分逐draw保持，固定m下线性，原条件期望零时才保持；协方差可改变，未学习随机参数。W可跨界查询，投影清除两段泄漏。守恒基于原确定性参考峰，不保证随机选峰后的原生PSD。两尺度可能产生接缝，保存分界、前/尾α、面积/泄漏/积分残差及原/新/校正接缝步长。保留外支撑/非正样本/原多峰及事件差异。

原M0和069 LEVELWARP实际重放为M0/LEVEL069，原12H指标报告必须精确069。候选SPLITWARP两头三H，合计18H。原6593/21047全人口、父行/权重/NaN/异常、FIT种子2026101839/1840、joint202610183900、A0/M0/005不变；AD/优化/有限差分/LP/DEV/确认0。所有H两头须原039全保护及wave>=2%/joint>=2%/三形态>=1T，之后另登记新MC确认。013T未硬件标定，程序完成和FIT好看不等于平均或随机响应接受。

070仍72NPZ、9profiles73791行272cells、12native/12obs各79116、4geometry/intrinsic/coupling各26372、156mean/random例子、128原EX/384native/128carrier。候选query/源目标knots须逐数组精确069；baseline旧科学数组精确069，reference列及intrinsic固定Q分组改LEVEL069，不是WARP068。新分段诊断只在候选geometry，原amplitude_scale列改为front_amplitude_scale与tail_amplitude_scale，分析脚本须适配。操作计数去除voltage_warp；baseline level_warp2/13186，candidate split_warp2/13186；其余原计数不变。

本地039/070、远端039/069CUDA/070CUDA均PASS，测试没有读TRAIN/CP。070CUDA独立mean1.77636e−15V/random5.05238e−17V/query+knot2.84217e−14sample，mean分段积分5.83561e−15Vns/random1.95156e−18Vns；恒等/反对称/线性/排列/批量/双峰/三角波半高点/空分段和非法结点显式失败通过。冻结{pkg['source_pins']}pins/{pkg['source_files']}manifest，源码包{pkg['archive_bytes']}B SHA{pkg['archive_sha256']}，deployment_01，无repair后缀。无科学代码或门槛变更。

远端rylan@100.82.231.102，{d['remote_root']}，run_01/control_01。派发{d['UTC']}，boot{d['boot_id']}，硬期限含宽限{deadline.isoformat()}；四身份{json.dumps(identities,ensure_ascii=False)}。完整argv/cwd见DISPATCH_RECEIPT/STARTUP_ASSESSMENT。派发N01无进程/锁空闲，GPUfree{d['GPU_free_MiB']}MiB。资源4CPU/GPU6GiB/RSS12GiB、内部780/外900+20秒、输出64MiB，单作业无重启，RESULT后仍查资源。

正确启动快照{a['verified_UTC']}：一次32秒有界请求取得首头9H，原6H精确069；全部6593候选查询及七结点对精确069且独立重建。query误差{a['query_max_error_samples']}sample/knot{a['knot_max_error_samples']}sample，mean分段残差{a['split_mean_integral_max_error_Vns']}Vns/random{a['split_random_integral_max_error_Vns']}Vns，seam元数据差{a['split_seam_metadata_max_error_V']}V；78mean公式{a['formula_max_error_V']}V/78random{a['coupling_formula_max_error_V']}V，6native/6obs39558行、5profiles、54W1（最大native差{a['W1_max_native_error']}在原浮点界内）通过，ADC整数码精确。RSS{a['RSS_bytes']}B/run{a['current_run_bytes']}B。三账本89jobs仅070active，正确启动后未再查询远端，此快照不是终态。

首头point_usefulness={json.dumps(first['point_usefulness'],ensure_ascii=False)}；decision={json.dumps(first['decision'],ensure_ascii=False)}。首头QPSD三H已过；tail fast=.018950010限.018507771，slow=.016971452限.017212357，common=.018746449限.018365082。虽已大幅恢复，仍按原界失败，不放宽到通过。两头最终结果尚未收集。

下次{next_wake}先一次有界身份/状态核对。适配069 COLLECT，使用本次真实root/四身份/config/deployment_01清单；仍运行按剩余预算安排下一次，不轮询/延预算。结束收齐exit/RESULT/COMPLETE、文件长度/hash、64MiB、真实最终深冻结与全部operation_bounds。

终态verifier基于070 REGISTER_STARTUP扩两头：原12H精确069，72NPZ，全13186查询/knots精确069且独立重建，所有分段不变量/接缝元数据，156mean/random、128原EX/384native、9profiles、108W1原界。query_error/amplitude_scale_error/knot_error/split_mean_error/split_random_error/seam_error均在头循环外初始化累计。baseline使用已核验069查询，候选独立NumPy分段α/投影，保留NaN/异常及固定Q参照例外。

收齐后分析全部H保护、接缝幅值与波形、峰迁移、原七坏行、三Q分组的宽度/前尾/PSD均值和方差/协方差。比较BASE054参考固定峰积分与实际native积分，分解小量残余tail失败是否集中在选峰/窗口或随机项；不能因差距小就当通过。完成报告/摘要/图像QA/闭合070，同次依据完整证据推进必要最小步骤；若所有FIT门槛真过，才另登记不重选端点的新MC确认。失败不盲扫强度、不直接全量随机训练、不扩大预算、不改目标。短步骤继续，长步骤正确启动后更新同heartbeat，保留连续推进责任。'''
(w/'FOLLOWUP_PROMPT.txt').write_text(prompt,encoding='utf-8');request=dict(mode='update',id=before['id'],kind=before['kind'],name=before['name'],prompt=prompt,rrule=f'FREQ=DAILY;BYHOUR={next_local.hour};BYMINUTE={next_local.minute};BYSECOND=0',status='ACTIVE',targetThreadId=before['target_thread_id'],destination='thread')
if 'notification_policy' in before:request['notificationPolicy']=before['notification_policy']
(w/'AUTOMATION_UPDATE_REQUEST.json').write_text(json.dumps(request,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
code=(r/'receipts/response069_20261018/deployment_01/VERIFY_FOLLOWUP.py').read_text().replace('response069','response070').replace("not j['response068']['active']","not j['response069']['active']").replace('prior87_job_records','prior88_job_records').replace('source459pins_and468manifest','source481pins_and490manifest')
(w/'VERIFY_FOLLOWUP.py').write_text(code)
print(json.dumps(dict(next_local_wake=next_wake,prompt_characters=len(prompt),id=request['id']),ensure_ascii=False))
