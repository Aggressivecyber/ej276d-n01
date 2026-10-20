from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,math

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
raw=json.loads((here/'WAIT_ESTIMATE_RAW.json').read_bytes())
early=json.loads((here/'STARTUP_RAW.json').read_bytes())
dispatch=json.loads((here/'DISPATCH_RECEIPT.json').read_bytes())
registration=json.loads((here/'REGISTRATION.json').read_bytes())
progress=raw['files']['run_01/PROGRESS.json'];counts=progress['execution_counts']
assert raw['boot_id']==dispatch['boot_id'] and raw['source_files_verified']==815
assert raw['config_sha256']==registration['config_sha256']
assert all(k not in raw['files'] for k in ('control_01/exit_receipt.json','run_01/FAILED.json','run_01/COMPLETE.json'))
assert progress['stage']=='original_precision_fix01_joint_Jacobian' and progress['seed']==0 and progress['column']==9
assert counts['predecessor_FD_columns_verified']==9 and counts['same_32draw_centers_verified']==5
by_pid={p['pid']:p for p in raw['processes']}
for p in early['processes']:assert by_pid[p['pid']]['start_ticks']==p['start_ticks']
bank_done=datetime.fromisoformat(early['files']['run_01/PROGRESS.json']['UTC'])
columns_done=datetime.fromisoformat(progress['UTC'])
elapsed=(columns_done-bank_done).total_seconds();per_column=elapsed/9
remaining_columns=108-9
remaining_candidate_calls=2*(3*4+1+2)
candidate_seconds=remaining_candidate_calls*per_column/6
second_head_setup=(bank_done-datetime.fromisoformat(dispatch['UTC'])).total_seconds()
remaining=remaining_columns*per_column+candidate_seconds+second_head_setup+20
estimated=columns_done+timedelta(seconds=remaining)
margin=.15*remaining
wake=(estimated+timedelta(seconds=margin)).replace(second=0,microsecond=0)+timedelta(minutes=1)
local=timezone(timedelta(hours=8))
estimate=dict(status='MEASURED_SINGLE_WAKE_ESTIMATE',UTC=datetime.now(timezone.utc).isoformat(),
    status_asof_UTC=raw['UTC'],progress_asof_UTC=progress['UTC'],first_nine_columns_seconds=elapsed,
    seconds_per_column=per_column,remaining_columns=remaining_columns,remaining_candidate_calls_upper=remaining_candidate_calls,
    remaining_candidate_seconds=candidate_seconds,second_head_setup_seconds=second_head_setup,
    output_and_solver_margin_seconds=20,estimated_remaining_seconds=remaining,estimated_finish_UTC=estimated.isoformat(),
    estimated_finish_local=estimated.astimezone(local).isoformat(),relative_wait_margin_fraction=.15,
    actual_wake_margin_seconds=(wake-estimated).total_seconds(),single_wake_UTC=wake.isoformat(),single_wake_local=wake.astimezone(local).isoformat(),
    prefix_replay=raw['fit_files']['0']['prefix_replay'],native_timing=progress['native_timing'],
    old_first_nine_seconds=546.638053,measured_first_nine_speedup=546.638053/elapsed,
    uncertainty='Projection uses the first nine full FD columns, upper-bound candidate calls and one measured setup interval. Shared-host load and later coordinates/solver cost may differ; existing budgets remain fixed.',
    budget_extended=False)
(here/'WAIT_ESTIMATE.json').write_text(json.dumps(estimate,indent=2)+'\n',encoding='utf-8',newline='\n')
prompt=f'''继续 E:/GAN/research/N01 的 JOINT_RESPONSE_01 五阶段。先完成所有当前可执行工作；仅长计算或其他必须等待才按本次实测速度及剩余工作估时，加余量安排本聊天单次苏醒，不用固定间隔轮询。醒来先删除或确认消费后的定时消失，没有等待事项不保留定时。全部由当前主助手完成，不使用子代理。数据、代码、日志仅E/D或原服务器；一次一个N01科学作业，不自动重启或延长原预算，历史失败保留。

本次单次接续时间{wake.astimezone(local).strftime('%Y-%m-%d %H:%M')}（UTC+8）。先读AGENTS.md、JOINT_RESPONSE_01_PLAN.md、PROGRAM.json、MASTER_EXECUTION_PLAN.json、jobs.json及docs/JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_FIX01_20261020.md。当前作业JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_FIX01_20261020；配置configs/joint_response_01_original_draw_precision_fix01_20261020.json，SHA256 678ebea3414d04da937e334c33957e67cc843496ff9ef01e1f8ead886442d386。服务器rylan@100.82.231.102，目录/home/rylan/projects/ej276d-n01/joint_response01_original_draw_precision_fix01_20261020；本地receipts/joint_response_01_20261020/original_draw_precision_fix01。00:52:17.186212 UTC派发；boot83c9f761-c124-42e1-89b8-5f45f0b3acc4；dispatcher148612及supervisor148613/ticks3133700，timeout148620/ticks3133701，science148622/ticks3133702。

最新快照00:53:49.081433Z核验815源文件/35 pins及原进程身份，无终止记录。进度00:53:12.041091Z：seed0已9/108参数差分列、27 FIT批列，首5组原4draw中心与嵌套库复现，首5组32draw中心和旧保存9列也通过。联合残差及参数数组逐值相同，native最大差8.326672684688674e-17；零求解/候选/参数更新。科学进程之后状态未知。RSS峰5771972608字节，低于12GiB；GPU分配上限4GiB。4CPU0-3、内部5400秒、外部5700秒加20秒、128MiB；近似内部截止10:22:17，外部最晚10:27:37，不延预算。

实测首9列44.409672秒，4.934408秒/列；余99列、最多30次高draw候选检查、第二头设置10.445207秒、20秒求解/输出余量，预计{estimated.astimezone(local).strftime('%H:%M:%S')}完成。{wake.astimezone(local).strftime('%H:%M')}苏醒留{estimate['actual_wake_margin_seconds']:.3f}秒余量，按实测剩余时间15%再向下一整分钟取整；不是固定间隔。若仍运行且身份一致，按实际剩余工作重新估一次单次苏醒。

本次原JointResponse01从M0零增量开始；固定已审计旧M0三FIT诊断的SOURCE/REAL行和五组joint/native投影。每角色1024父，原4draw逐值保留z及三种噪声，追加独立28draw成为32768行；REAL固定4096行。FIT0/FIT1/FIT2堆叠2448维完整J，27个native约束；54参数中央差分0.004，ridge.02，半径.0125，最多3候选/头、拒绝FIT割线修正并半径减半。原全部FIT和CHECK至少1%joint收益、宏块1.15+2/4096、TR均值.08/p99.25、flag增量max(4,.005n)、wave/QPSD≤1.01和noise≤1.10不变。首个通过候选只作一次FRESH检查；所有候选均零提交更新。旧4draw未提交候选只在32draw CHECK/FRESH作配对参照。FRESH沿用已暴露拟合角色行/投影，仅追加28新draw；不是全流程未见验证。128 SOURCE和2099 REAL报告角色均排除。

FIX01只把均匀无显式权重且8:1样本比的W1换成精确次序统计量计算。进程内显式绑定run_response036和n01.losses008_v2的W1符号；其余权重/比例保留原函数。没有在数据层复制REAL、改分箱支持、窗口、统计分母或科学门。原模型及loss文件字节不变。工程测试六组+SciPy参照最大差4.552e-15；带权值和权重梯度逐值一致。旧作业首9列546.638秒，单FD预计6559.657秒超原5400秒预算，已核对身份后SIGTERM停止：exit-15，9列5中心及256M0零增量行审计通过，末次快照后0—9列未知费用区间保留。旧作业不重启、不延长；FIX01重新从M0执行一次单独登记的相同科学诊断。

本次最多108参数列/324FIT批列，6联合求解和候选，4割线修正，2FRESH新候选检查及4旧候选检查。20随机库/噪声调用327680行、983040扩展行、2623488正态数；700生成调用22585600行（含256零增量行），2094观察调用67756032行，native76333056行。6检查点、1TRAIN/1PCA、A0 52calls6593rows、M0 106calls13442rows；5joint/5native图，REAL观察21047行1次，FIT特征24576行及CHECK/FRESH16384行。新增保存证据比较5中心9列，不新增生成。各未使用候选按4高draw调用及其实际生成/观察/native行数扣除；没有被接受候选时FRESH检查不执行。REAL训练、DEV、AD、classifier0。

醒来一次有界查询REMOTE_WAIT_ESTIMATE_REQUEST.py，核验boot/PID/ticks/815源及exit/COMPLETE/FAILED。终止后已有COLLECT.py（REMOTE_COLLECT.py+scp+哈希核验，解至terminal_01/collected）和AUDIT_COMPLETE.py，后者当前仅AST通过尚未跑终止数据；需独立复算矩阵、全部实际候选风险、割线、完整SLSQP、TR摘要门、原4draw中心/父REAL索引、原32draw五中心九列及实际费用。独立SLSQP审计核心已对四历史方向复算，参数差<2e-17。保存向量审计不等于原始波形重放；TR RMS/flag及嵌套噪声逐值校验依赖原生记录。若完整结果出现零方向，检查审计对零方向分支的范围，不能仅用宽松计数掩盖未复核求解。失败保存完整前缀与未知执行区间，不自动重启。

本次08:15接续已经闭合C1直接原教师高精度恢复：exit0，1007.64秒，785源/125pins/46输出；432FD/8方向/20候选/4更新，36硬门通过16/18精度门，seed1条件均值5.7628%/5.7221%>5%。随后冻结五端点、两教师家族、每头四独立全6465父×32draw库，93.10秒完成零更新目标排序诊断；796源/136pins/12输出240残差向量复算。原目标在24配对视图全部偏好条件较差C1端点，matched-C1目标全部反转；不能继续同一跨家族目标或用已暴露真值调Q随机尺度。已有C1导数替代原生成器导数检查也失败。

原JointResponse01教师恢复已有36硬+18精度门通过记录（receipts/joint_response_01_20261019/kernel_precision_confirmation_01/TERMINAL_REVIEW.json）；据此回到原生成器检验随机精度。旧REAL三拟合批两头各24/96改善门失败，门不变。当前两头32draw均过原局部FIT/CHECK/FRESH后，结合所有必要物理/人口对照，登记一个最小有界后续联合校准；未通过则先定位高draw J/native/宏块差异，不能直接扩网或加迭代。阶段4所有最终门通过才进入阶段5同父随机核、输入干预、人口/观察敏感性、新C2ST和全流程留出。A0/M0/005已暴露全TRAIN，当前只称冻结系统后验研究评估。

更新三台账及说明，同步代码、配置、汇总证据至原GitHub；私有波形、逐行身份、NPZ、权重不上传。状态不变或无行动时保持安静，仅实质进展、完成、失败或需要用户操作时通知。'''
request=dict(mode='create',kind='heartbeat',destination='thread',name='N01 原生成器精度诊断接续',
    targetThreadId='01a12bad-8839-7522-89a5-f028b107d7ff',status='ACTIVE',
    rrule=f'FREQ=DAILY;COUNT=1;BYHOUR={wake.astimezone(local).hour};BYMINUTE={wake.astimezone(local).minute};BYSECOND=0',prompt=prompt)
(here/'AUTOMATION_CREATE_REQUEST.json').write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(estimate))
