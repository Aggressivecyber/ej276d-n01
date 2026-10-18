from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy
w=Path(__file__).parent;r=w.parents[2];out=r/'reports/response070_20261018';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();now=datetime.now(timezone.utc).isoformat()
v=read(w/'ARTIFACT_VERIFICATION.json');a=read(w/'SPLIT_WARP_ANALYSIS.json');pa=read(w/'SPLIT_WARP_TOPOLOGY.json');ta=read(w/'TAIL_AND_CONDITIONAL_PSD_AUDIT.json');nt=read(w/'NATIVE_TAIL_DECOMPOSITION.json');res=read(w/'collected/run_01/RESULT.json');ex=read(w/'collected/control_01/exit_receipt.json');c=read(w/'COLLECTION_VERIFICATION.json')
assert v['status']=='PASS' and res['stage_outcome']=='CLOSED_NEGATIVE_SPLIT_AREA_CROSSING_LEVEL_WARP'
next_design='071 frozen070 replay and all-population native tail/peak decomposition, no new response candidate. Replay all18H exactly070, reconstruct BASE054 observations on the same six FIT conditions, and separate paired native Q/tail changes into same-new-window mean/random/ADC changes and BASE054 window shift. Save intermediate deterministic, random, electronics and ADC peak indices and peak competition contributions. Preserve039 gates and070 negative outcome; no confirmation, no parameter fitting or strength scan. Use the result to choose a justified minimal stochastic-timing or morphology correction.'
rows='\n'.join(f'| {x["seed"]} | {x["hypothesis"]} | {100*x["usefulness"]["wave_relative_gain"]:.3f}% | {100*x["usefulness"]["joint_relative_gain"]:.3f}% | {x["usefulness"]["morphology_mean_T_gain"]:.6f} | {x["MIX"]["QPSD"]:.6f} | {x["MIX"]["tail_Vns"]:.6f} | {", ".join(z.split("/",1)[1] for z in x["violations"]) or "none"} |' for x in a['metrics'])
width=next(x for x in a['features'] if x['seed']==0 and x['case']=='SPLITWARP' and x['kind']=='fast' and x['group']=='fixed_BASE' and x['field']=='FWHM_ns');rr=next(x for x in a['REAL_features'] if x['field']=='FWHM_ns')
ratios=[z['groups']['different_native_peak']['sum_abs_delta_Vns']/(z['groups']['different_native_peak']['sum_abs_delta_Vns']+z['groups']['same_native_peak']['sum_abs_delta_Vns']) for z in nt['population']]
text=f'''# 070：分段面积恢复，原生选峰相关的尾部差异仍未通过

状态 CLOSED_NEGATIVE_SPLIT_AREA_CROSSING_LEVEL_WARP。两头、三种噪声假设的执行与复核完成；五个条件的tail距离失败，第二头fast/common_level还失败QPSD。全部点目标通过，但均值阶段未接受，完整随机响应目标仍开放，未开新确认库。

| 头 | H | wave收益/M0 | joint收益/M0 | 三形态T收益 | QPSD | tail W1/V ns | 原保护失败 |
|---|---|---|---|---|---|---|---|
{rows}

原069全部六个QPSD/tail失败现已大幅缩小。当前首头slow的全部保护通过，其他五个条件仍失败，按原039界限保留，不因差距小而放宽。六H波形收益29.947%–32.603%，联合形态57.894%–58.497%，三形态平均3.6813–3.7084T。峰高/前沿/宽度保护均通过。

确定性前尾积分恢复到BASE054。6593个确定性峰逐行保持；新旧参考尾部最大差8.88e-16V ns，所有行在1e-12容差内。原生tail相对BASE054的均值变化仍为+0.001250至+0.001816V ns。因此，剩余差异来自参考积分以外的随机响应与共同读出组合。它不能从一个残差项直接归因于某一种噪声。

全人口配对标量表显示，相对BASE054的原生峰不同事件为981–1046/6593，约15%；这些行占绝对尾部差值总和的{100*min(ratios):.2f}%–{100*max(ratios):.2f}%。两者都选在确定性均值峰时，5337–5420行的平均绝对tail差约9.1e-5V ns，最大4.88e-4V ns，与分段面积守恒后保留的ADC离散误差相符。这个分组定位关联，尚不是全人口因果分解。

作为标量归因诊断，将“峰不同”组的tail值替换为相同父行BASE054值，六H W1变为0.018241/0.016643/0.018066/0.020150/0.018612/0.020497V ns，均在原各自tail界内；替换“峰相同”组几乎不改失败。这只定位保存的标量差异，未构成可执行响应模型，也不代表任何候选通过。78个已保存fast完整例子进一步精确分解为“相同新窗口中的波形变化”与“BASE054波形的窗口移动”两项，并分离参考窗口的均值、随机和ADC差。完整人口同样的分解仍需071一次有界冻结重放。

接缝需要保留审查。前段幅值系数范围0.748598–1.674281，尾段0.660521–2.210132；p+30的确定性修正接缝为−29.909至+73.032mV。由于不同原生峰引起的差异已占主要份额，当前证据不足以断言单独平滑接缝能解决尾部保护。未以接缝小或总面积正确代替共同读出评价。

首头fast固定LEVEL069输出Q八箱FWHM均值为{[round(x,4) for x in width['bin_means']]}ns，REAL为{[round(x,4) for x in rr['bin_means']]}。宽度总/箱内/箱间方差为{width['variance']:.6f}/{width['within']:.6f}/{width['between']:.6f}ns²，对REAL {rr['variance']:.6f}/{rr['within']:.6f}/{rr['between']:.6f}。070保留069恢复的宽度关系；低Q峰前与尾部形状仍有残差，高Q箱内波形变化仍偏大。人口条件方差包含源事件差异，不能当成同一输入的随机核方差。

全部六H FWHM未定义为0，原066七个失败父行每头保持有定义；rise未定义36–40/6593，均通过原保护。全部七类observation flags为0。均值到随机signal移峰392/424行；均值到native为578–675行；相对069仅25–36行移峰。保留inputQ、固定LEVEL069输出Q、各自输出Q的所有分组，SOURCE与REAL仍不逐事件配对，Q不是能量真值。

下一步071只做冻结诊断：完整重放070的18H并逐报告一致，补算相同输入/随机库/电子噪声下的BASE054观察；全6593行逐H保存选峰阶段、Q/tail的均值/随机/量化/裁窗分解与峰竞争贡献。它将判别误差主要在随机项耦合还是最终噪声选峰，以及是否存在有证据支持的最小修正。没有新端点、拟合、强度网格或确认库，不预设“改变随机项一定成功”。

执行验收：exit0，原生{res['elapsed_seconds']:.6f}秒，峰RSS{ex['peak_rss_bytes']}B。{c['file_count']}文件{c['uncompressed_bytes']}B逐长度/SHA核验，run{c['run_bytes']}B低于64MiB。481冻结引用/490部署源码一致；18H精确计数与真实最终A0/M0/005深冻结通过。原M0/LEVEL069的12H完整指标报告精确069，原科学数组不变，固定Q参照例外单列。

72NPZ、9profiles/73791行/272组视图、geometry/intrinsic/coupling各26372行、12native/12obs各79116行核验。13186查询和七结点对精确069且独立重建；查询误差{v['query_max_error_samples']:.3g}sample、结点误差0。均值/随机分段残差最大{v['split_mean_integral_max_error_Vns']:.3g}/{v['split_random_integral_max_error_Vns']:.3g}V ns。156mean/random例子、128原EX、384native和128carrier检查通过；ADC整数码精确。108W1用SciPy及整数CDF独立复核，最大native差{v['W1_max_native_error']:.3g}，沿用原浮点累计界。未保存全人口完整波形，未离线重算完整wave/QPSD；013T仍为人工扰动研究尺度。

![条件统计](DIAGNOSTIC.png)

![尾部与条件PSD](TAIL_DIAGNOSTIC.png)

![原生尾部分解与接缝](NATIVE_TAIL_DIAGNOSTIC.png)

![旧失败事件](PEAK_EXAMPLES.png)
'''
(out/'REPORT.md').write_text(text,encoding='utf-8');(out/'SUMMARY.json').write_text(json.dumps(dict(stage=res['stage_outcome'],scientific_goal_achieved=False,verification=v,collection=c,metrics=a['metrics'],topology=pa,tail_population_audit=ta['population'],BASE054_native_reference=ta['BASE054_native_reference'],native_tail_decomposition=nt,next_design=next_design),indent=2)+'\n')
(w/'VISUAL_QA.json').write_text(json.dumps(dict(UTC=now,images=[dict(path=(out/name).relative_to(r).as_posix(),sha256=sha(out/name),visually_inspected=True,panels=panels,all_panels_legible=True,no_clipped_labels=True) for name,panels in [('DIAGNOSTIC.png',6),('PEAK_EXAMPLES.png',6),('TAIL_DIAGNOSTIC.png',3),('NATIVE_TAIL_DIAGNOSTIC.png',3)]]),indent=2)+'\n')
review=dict(v,ended_UTC=ex['ended_utc'],collection=c,report=(out/'REPORT.md').relative_to(r).as_posix(),report_sha256=sha(out/'REPORT.md'),analysis_sha256=sha(w/'SPLIT_WARP_ANALYSIS.json'),topology_sha256=sha(w/'SPLIT_WARP_TOPOLOGY.json'),tail_audit_sha256=sha(w/'TAIL_AND_CONDITIONAL_PSD_AUDIT.json'),native_tail_decomposition_sha256=sha(w/'NATIVE_TAIL_DECOMPOSITION.json'),next_design=next_design,full_scientific_goal_achieved=False)
(w/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');before=w/'before_closure';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 path=r/name;(before/name).write_bytes(path.read_bytes());d=read(path);job=d['response070'];assert job['active'] and job['stage']=='RUNNING_STARTUP_VERIFIED'
 job.update(active=False,stage=res['stage_outcome'],ended_UTC=ex['ended_utc'],closed_UTC=now,terminal_review=(w/'TERMINAL_REVIEW.json').relative_to(r).as_posix(),terminal_review_sha256=sha(w/'TERMINAL_REVIEW.json'),report=review['report'],report_sha256=review['report_sha256'],execution_counts=res['execution_counts'],native_exit_code=0,scientific_goal_achieved=False)
 d.update(status='response070_closed_negative_split_area_warp',current_stage='response071_native_peak_tail_decomposition_preparation',active_scientific_jobs=0,active_job_id=None,updated_UTC=now,current_action='070 reference areas restored; remaining native tail discrepancy concentrates in events whose peak differs from BASE054.',next_action=next_design)
 d['response_goal_20261018'].update(stage_status='preparing_bounded_native_peak_tail_decomposition',current_stage=d['current_stage'],next_action=d['next_action'],updated_UTC=now)
 if name=='jobs.json':assert len(d['jobs'])==89;d['jobs'][-1]=copy.deepcopy(job)
 path.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
assert read(before/'jobs.json')['jobs'][:-1]==read(r/'jobs.json')['jobs'][:-1]
receipt=dict(UTC=now,response070_stage=res['stage_outcome'],active_scientific_jobs=0,prior88_job_records_unchanged=True,report_sha256=review['report_sha256'],terminal_review_sha256=sha(w/'TERMINAL_REVIEW.json'),root_only=True,scientific_goal_achieved=False,continue_same_turn=True)
(w/'CLOSURE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
