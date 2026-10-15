# 019 原生监督：只读复用 017

本目录只准备启动方案；未 SSH、部署、启动、提交或修改共享登记。018 数组结果的既有独立核实完成并由 ROOT 放行后，ROOT 冻结 019 源码、CFG、manifest、ops SHA，再执行单次原生启动。

`REMOTE_SUPERVISOR.py` 是顶层 bootstrap，没有新函数、类、GPU 循环或 controller。它只读加载旧017 `operations()`，显式绑定019常量及 `run_wp4_019.validate_settings019(p, *, frozen=True, driver_source=None)`，直接调用原 `supervise(config_sha, lock_fd)`。GPU/RSS/owned-tree、TERM/KILL、资源限值、atomic 保存与 durable journal 仍由原成熟代码执行。

## 冻结接口

- `run_id=wp4_019_20261016`；owned root `/home/rylan/projects/ej276d-n01/wp4_019_20261016`。
- CLI：`python -B REMOTE_SUPERVISOR.py --config <root>/wp4_019_config.json --config-sha256 <冻结RAW SHA>`。
- `WP4_019_SOURCE_MANIFEST.json` 的 `config_sha256` 绑定 CFG 原字节，`files` 与 `CFG.source_files` 集合及 bytes/SHA 一致；ops 本身以源或合同绑定并由 ROOT 固定 SHA。
- CFG 绑定 `inputs/contracts/protection_files/allowed_input_roots`；A/B/Q 只读旧017 cache，005 CP 只加载一次，无 A/M0 CP、NN 预计算、SOURCE/REAL/DEV 解码。原017/018源码、数组、门和结论保持。
- 当前工厂：两 T1 unpaired seed 各 2000 G/AD，总 4000；CPU12、GPU10 GiB、进程树 RSS8 GiB、batch wall21600 s、output5120 MiB、trial wall7200 s、collection128 MiB。最终数值只以已审冻结 CFG 与预算工厂为准；不恢复旧 stage2，不执行 T2。
- supervisor 只获取既有父根 `.n01_scientific_dispatch.lock`，旧监督函数在结束时关闭 fd；没有自动重启。preflight 复用旧 GPU/RAM/CPU 检查，并执行 CFG 中 40 GiB 最低可用盘要求。

旧源码只读依赖（路径均相对旧017 root，必须作为保护绑定保留）：

| 文件 | bytes | RAW SHA256 |
| --- | ---: | --- |
| scripts/wp4_017_controller.py | 31507 | eb15f3ba35844436f9b5ff3879332bd19046fcfd4f1fffe8459db2505333c38c |
| scripts/wp2_controller.py | 50989 | 3480825e94e954a46c2c633e5df6ddef514e2b7def1332bb2e7688e78ae041a3 |
| scripts/mechanism012_ops.py | 62946 | 98a18eef018beebff82565da9ed5b31208cf2bff6db6bff0c492eff9bf2a7816 |
| scripts/wp1_ops.py | 12872 | f8ddbbdb2f2f6df033e964f27bf03dcbda07055cc7378775591ef69b9caf85b5 |
| scripts/supervise_mechanism012.py | 6505 | 43b6fffd076a1e6d57db67e85e433e71d3f2cda91f3a3d5fe31f0fcb430486a4 |

这些原语的 launch/exit schema 前缀仍是 `n01/wp4-017-*`，仅表示继承的记录格式。实际科学身份必须共同核对 **019 run_id、冻结 CFG/SHA、pins、owned cwd、实际命令和 PID/start_ticks/boot**；START_RECEIPT 会明确这一点。不得调用旧017 `native_start/first_valid/complete_valid` 来判019，它们含旧科学条件。

## ROOT 部署和启动建议

沿用[018 已审结构 argv 传输方式](../../wp4_018_20261016/execution_01/DEPLOY_LAUNCH.md)，但019源包成员数取实际冻结 manifest，不能硬套018的6文件。新 root 与 staging 均须 fresh，禁止覆盖旧017/018或既有019；上传后核对归档全SHA、每个允许 regular member 的 bytes/SHA、CFG/MF RAW SHA、输入清单与 ops SHA，再创建 owned root。只传公开源码/合同/控制文件，原科学输入保留原路径。

ROOT 完成登记与冻结提交后，单次启动使用结构参数数组；以下 SHA 填最终冻结回执值，不把 JSON 当 shell quoting：

```python
import shlex, subprocess
HOST = 'rylan@100.82.231.102'
PY = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
REMOTE = '/home/rylan/projects/ej276d-n01/wp4_019_20261016'
ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST]
cfg_sha, manifest_sha, ops_sha = '<CFG RAW SHA>', '<MF RAW SHA>', '<独审ops SHA>'
launch = r'''
from pathlib import Path
import hashlib,json,subprocess,sys
r=Path(sys.argv[1]); cfg_sha,manifest_sha,ops_sha=sys.argv[2:]
for name,expected in [('wp4_019_config.json',cfg_sha),
    ('WP4_019_SOURCE_MANIFEST.json',manifest_sha),('REMOTE_SUPERVISOR.py',ops_sha)]:
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected
command=[sys.executable,'-B',str(r/'REMOTE_SUPERVISOR.py'),'--config',
    str(r/'wp4_019_config.json'),'--config-sha256',cfg_sha]
with (r/'supervisor_console.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=r,stdin=subprocess.DEVNULL,stdout=log,
        stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(supervisor_pid=child.pid,command=command,config_sha256=cfg_sha)))
'''
result = subprocess.run([*ssh, shlex.join([PY, '-B', '-', REMOTE,
    cfg_sha, manifest_sha, ops_sha])], input=launch.encode(), capture_output=True)
# 将 result.stdout/result.stderr 原字节与 returncode 保存到本次 E/D 回执，再检查返回码。
result.check_returncode()
```

bootstrap 保存019 `DISPATCH_ATTEMPT.json` 和 `START_RECEIPT.json`；原监督函数保存 `runs/primary/supervisor/{launch,resource_status,exit_receipt}.json`、`runs/primary/console.log`，只执行一次019 driver CLI `--config --config-sha256 --output --device cuda:0`。

## 首次实际更新后的停止点

ROOT 在同一次新鲜快照内保存019 `INIT_PROOF.json`、`FIRST_UPDATE_PROOF.json`、`status.json`、durable enter/completed/inflight 与当前 `/proc` / owned-tree/GPU/RSS/output/elapsed。必须看到首试 `T1_unpaired_seed0` 的 `theta_a/theta_b` 两梯度和两实际更新均非零、G/AD=1、005 CP=1、零 A/M0 CP/NN 前算、缓存与005冻结身份一致，以及当前 boot/PID/start_ticks/父子关系/cmdline/cwd/CPU12和资源限值健康。不得用历史首G1替代当前健康，也不得套旧017的CP=3及A前算验收。

通过后只设一次未来苏醒并休息；OS内原监督循环继续治理资源，agent不持续轮询。终结时按019正式预算、native exit、COMPLETE/FAILED、实际 trial 集合和 durable 计数记账；native0与科学门分别报告。精选回收严格不超过128 MiB，head CP与完整 inventory/SHA 留据，大数组保留远端；若完整日志本身超过精选预算，先记录实际大小和取舍交 ROOT，不偷偷丢日志或扩大预算。
