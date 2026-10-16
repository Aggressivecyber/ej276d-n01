# 020 恢复执行入口（仅准备）

`REMOTE_SUPERVISOR.py` 由019已审4265 B bootstrap作常量/validator绑定替换得到。仍直接调用旧017 `operations().supervise(config_sha, lock_fd)`；没有新增循环、函数、类、controller或盲重启。本阶段未SSH、部署、启动、提交或改共享metadata、旧源码及失败019。

## 授权与本次范围

[用户新授权](../authorization_01/USER_AUTHORIZATION.json) RAW SHA256 为 `d1dcdd2002cf52c792aa307e14cc11fb3b1ad25d035ff0cd46a8bb84cc0040fa`（1874 B），允许ROOT后续自主工程修复、重新执行及必要新增计算，无需逐次再询问用户。**本入口每个native尝试仍只运行一次**：失败如实留据退出，后续修复由ROOT依据证据另行冻结、登记和记账，禁止无界重试或覆盖原失败。

本020范围：005 CP加载1次、恢复019的seed0 final CP 1次，总CP=2；seed0只补原final评价，新增seed0 G/AD=0、无seed0 optimizer与parent sampler。随后seed1只训练原定2000 G/AD一次，并完成原final评价。旧seed0的2000 G保留在历史累计，不重新计入本020新增G。科学门继续使用已审018规则，不降低门或擅自扩展试验。

## 绑定与资源

- owned root：`/home/rylan/projects/ej276d-n01/wp4_020_20261016`；driver：`scripts/run_wp4_020.py`。
- CFG：`wp4_020_config.json`；MF：`WP4_020_SOURCE_MANIFEST.json`；输入清单：`WP4_020_SAVED_INPUTS.json`。
- validator：纯stdlib `validate_settings020(p, *, frozen=True, driver_source=None)`，由020源工厂与精确预算检查冻结CFG。
- MF的 `config_sha256` 绑定CFG原字节，`files` 与 `CFG.source_files` 的成员、bytes/SHA一致；ops单独RAW SHA固定。新输入与保护路径只由冻结CFG/manifest给出。
- 五个旧017 ops依赖与RAW SHA完全沿用[019已审矩阵](../../wp4_019_20261016/execution_01/DEPLOY_LAUNCH.md)，仍只读保护；A/B/Q读旧017 cache，无A/M0 CP或NN预计算。
- batch wall21600 s、output5120 MiB、shared1024 MiB、trial wall7200 s/output1536 MiB、collection128 MiB、GPU10 GiB、进程树RSS8 GiB、CPU12；以020正式工厂和冻结CFG为准。
- 既有父根 `.n01_scientific_dispatch.lock`、GPU/CPU/RAM preflight、CFG最低40 GiB磁盘、owned tree、资源终止和atomic/durable保存原语均不重写。

继承的supervisor launch/exit schema前缀仍是 `n01/wp4-017-*`，仅为记录格式。实际身份取020 `run_id`、CFG/MF/ops pins、owned cwd/命令与PID/start_ticks/boot。不得调用旧017科学 `first_valid/complete_valid`，也不得套019的CP=1或旧017的CP=3条件。

## ROOT 单次部署和启动

ROOT完成这一轮既有独审、020登记、冻结提交与实际服务器preflight后，沿用[019结构argv部署约束](../../wp4_019_20261016/execution_01/DEPLOY_LAUNCH.md)。新root/staging须fresh，不覆盖旧017/018/019或任何已有020；核对整个公开源包、全部允许regular members、CFG/MF、输入清单及ops的完整bytes/SHA，原科学输入保留原路径。以下仅是ROOT执行建议，SHA取最终冻结回执：

```python
import shlex, subprocess
HOST = 'rylan@100.82.231.102'
PY = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
REMOTE = '/home/rylan/projects/ej276d-n01/wp4_020_20261016'
ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST]
cfg_sha, manifest_sha, ops_sha = '<CFG RAW SHA>', '<MF RAW SHA>', '<最终ops SHA>'
launch = r'''
from pathlib import Path
import hashlib,json,subprocess,sys
r=Path(sys.argv[1]); cfg_sha,manifest_sha,ops_sha=sys.argv[2:]
for name,expected in [('wp4_020_config.json',cfg_sha),
    ('WP4_020_SOURCE_MANIFEST.json',manifest_sha),('REMOTE_SUPERVISOR.py',ops_sha)]:
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected
command=[sys.executable,'-B',str(r/'REMOTE_SUPERVISOR.py'),'--config',
    str(r/'wp4_020_config.json'),'--config-sha256',cfg_sha]
with (r/'supervisor_console.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=r,stdin=subprocess.DEVNULL,stdout=log,
        stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(supervisor_pid=child.pid,command=command,config_sha256=cfg_sha)))
'''
result = subprocess.run([*ssh, shlex.join([PY, '-B', '-', REMOTE,
    cfg_sha, manifest_sha, ops_sha])], input=launch.encode(), capture_output=True)
# ROOT将stdout/stderr原字节和returncode留在本次E/D回执，再检查返回码。
result.check_returncode()
```

实际driver CLI仍为 `--config --config-sha256 --output <root>/runs/primary/artifacts --device cuda:0`。bootstrap只写新020 `DISPATCH_ATTEMPT.json` / `START_RECEIPT.json`，原监督函数写 `runs/primary/supervisor/{launch,resource_status,exit_receipt}.json` 与 `runs/primary/console.log`。

## 分阶段证明与苏醒停止点

**阶段一：恢复seed0并补final评价。** ROOT在一次新鲜快照中核对 `INIT_PROOF.json` 与 `RESTORED_SEED0_PROOF.json`：020身份；来源run019、trial `T1_unpaired_seed0`、source G=2000；来源CP的完整SHA与冻结输入一致；实际恢复仅 `theta_a/theta_b`；005 CP1 + restored CP1 = 总CP2；seed0新增G/AD=0、optimizer=false、parent samplers=0；缓存manifest与005冻结身份一致。

`INIT_PROOF.json` 在恢复前写出，记录005 CP1、restored CP0；总CP2由随后 `RESTORED_SEED0_PROOF.json` 证明，不能把恢复前的INIT误作本020最终加载数。

同时保存实际已完成多批硬观测/诊断的durable journal计数、当前status/phase及inflight状态，检查没有FAILED/异常退出；在同一快照中核对当前父子PID/start_ticks/boot/parent关系、cmdline/cwd、worker CPU12、owned-tree GPU/RSS/output/elapsed均在冻结限值内。这证明恢复后的**评价任务正确启动**。此阶段新增G/AD仍为0属正常；seed0完整final评价会先消耗421952个student evaluation rows，不能写成seed1已经训练。ROOT可据上述实证设置一次合理未来苏醒并休息，无需持续轮询等待seed1G1。

**阶段二：seed1真实训练。** 后续苏醒或终点必须核查新的 `FIRST_UPDATE_PROOF.json`（以及trial副本）：020身份、trial `T1_unpaired_seed1`，seed1首次G/AD=1，`theta_a/theta_b`两梯度和两实际参数delta均非零；同时确认RESTORED_SEED0证明仍在、总CP2、无seed0新增训练、缓存/005冻结一致。首次更新必须与该次当前owned进程和资源健康共同验证，历史G1不能替代当前健康。若已终结，则按真实native0、完整终结计数与身份链核查，而不要求已退出进程仍存活。

两阶段证据分别记录，不互相替代。终结时核对seed0仅恢复/评价、seed1只2000新G/AD、两个final评价、精确预算与durable计数，再报告科学门。native0与科学接受保持区分；精选回收不超过128 MiB，大数组保留远端，完整inventory与遗漏列表留据。
