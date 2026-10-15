# 018 原生执行建议（尚未部署或启动）

ROOT 在本轮独审结束、冻结 CFG/6 文件源包/75 输入清单并登记后执行。下面的本地 Python 变量取实际冻结回执；所有日志与回执保存在 E/D。这里没有执行 SSH。

监督脚本直接 `Popen` driver，并用 `wait(timeout=7200)` 计时，不套 GNU timeout。它记录直接子进程的 native returncode（信号终结为负数），超时只 TERM 自己新建的进程组，20 秒后仍未退出才 KILL；没有重启。driver 的 AS=6 GiB、CPU soft/hard=21590/21600 s 与 phase wall/output guard 沿用已审实现。监督进程额外记录 getrusage、最终输出大小及超界 FAILED。

## 本地参数与独占 staging

```python
from pathlib import Path
import hashlib, json, shlex, subprocess
HOST = 'rylan@100.82.231.102'
PY = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
REMOTE = '/home/rylan/projects/ej276d-n01/wp4_018_20261016'
STAGE = '/home/rylan/projects/ej276d-n01/.wp4_018_incoming_20261016_01'
ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST]
packet = Path(r'<ROOT实际冻结源包路径>')
cfg = Path(r'E:\GAN\research\N01\wp4_018_config.json')
ops = Path(r'E:\GAN\research\N01\receipts\wp4_018_20261016\execution_01\REMOTE_SUPERVISOR.py')
packet_sha, cfg_sha, ops_sha = '<冻结包SHA>', '<冻结CFG SHA>', '<独审后ops SHA>'
p = json.loads(cfg.read_bytes())
inputs = cfg.parent/p['saved_input_manifest']
assert inputs.name == p['saved_input_manifest']
assert hashlib.sha256(cfg.read_bytes()).hexdigest() == cfg_sha
assert hashlib.sha256(packet.read_bytes()).hexdigest() == packet_sha
assert hashlib.sha256(ops.read_bytes()).hexdigest() == ops_sha
prepare = "from pathlib import Path; import sys; r=Path(sys.argv[1]); s=Path(sys.argv[2]); assert not r.exists(), 'preserve existing018'; s.mkdir(exist_ok=False); print(s)"
subprocess.run([*ssh, shlex.join([PY, '-B', '-', REMOTE, STAGE])], input=prepare.encode(), check=True)
subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
    str(packet), str(cfg), str(inputs), str(ops), HOST+':'+STAGE+'/'], check=True)
```

本地进程以 argv 数组调用；SSH 唯一远端命令用 `shlex.join` 编码。JSON 只作为文件内容，不用 `json.dumps` 代替 shell quoting。

## 验证上传边界并创建 fresh owned root

```python
install = r'''
from pathlib import Path
import hashlib,json,sys,tarfile
s,r=Path(sys.argv[1]),Path(sys.argv[2])
archive_name,packet_sha,cfg_sha,ops_sha=sys.argv[3:]
raw=(s/'wp4_018_config.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==cfg_sha
p=json.loads(raw); assert p['run_id']==r.name and p['status']=='frozen'
assert len(p['source_files'])==6
manifest_name=p['saved_input_manifest']; manifest=(s/manifest_name).read_bytes()
assert hashlib.sha256(manifest).hexdigest()==p['saved_input_manifest_sha256']
ops=(s/'REMOTE_SUPERVISOR.py').read_bytes()
assert hashlib.sha256(ops).hexdigest()==ops_sha
assert hashlib.sha256((s/archive_name).read_bytes()).hexdigest()==packet_sha
payload={}
with tarfile.open(s/archive_name,'r:*') as archive:
    members=archive.getmembers()
    assert sorted(m.name for m in members)==sorted(p['source_files'])
    for member in members:
        assert member.isfile()
        data=archive.extractfile(member).read(); row=p['source_files'][member.name]
        assert str(r/member.name)==row['path']
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        payload[member.name]=data
payload.update({'wp4_018_config.json':raw,manifest_name:manifest,'REMOTE_SUPERVISOR.py':ops})
r.mkdir(exist_ok=False)
for name,data in payload.items():
    path=r/name; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream: stream.write(data)
receipt=dict(run_id=r.name,config_sha256=cfg_sha,source_package_sha256=packet_sha,
    supervisor_sha256=ops_sha,source_files=6,scientific_started=False)
with (r/'REMOTE_INSTALL_RECEIPT.json').open('x') as stream: json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
'''
subprocess.run([*ssh, shlex.join([PY, '-B', '-', STAGE, REMOTE, packet.name,
    packet_sha, cfg_sha, ops_sha])], input=install.encode(), check=True)
```

75 个原输入不复制、不解码、不改写；driver 在正式运行内核对该清单与原字节 SHA。源包只能有 CFG 绑定的 6 个 regular member，ops 单独绑定 SHA。已有远端 root 或 staging 会直接停止，不覆盖旧017或既有018。

## 单次启动

```python
launch = r'''
from pathlib import Path
import hashlib,json,os,subprocess,sys
r=Path(sys.argv[1]); cfg_sha,ops_sha=sys.argv[2:]
assert hashlib.sha256((r/'wp4_018_config.json').read_bytes()).hexdigest()==cfg_sha
assert hashlib.sha256((r/'REMOTE_SUPERVISOR.py').read_bytes()).hexdigest()==ops_sha
command=[sys.executable,'-B',str(r/'REMOTE_SUPERVISOR.py'),'--config',
    str(r/'wp4_018_config.json'),'--config-sha256',cfg_sha]
with (r/'supervisor_console.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=r,stdin=subprocess.DEVNULL,stdout=log,
        stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(supervisor_pid=child.pid,command=command,config_sha256=cfg_sha)))
'''
subprocess.run([*ssh, shlex.join([PY, '-B', '-', REMOTE, cfg_sha, ops_sha])],
    input=launch.encode(), check=True)
```

启动只有一次。原生 supervisor 获取父根 `.n01_scientific_dispatch.lock`，把同一锁 fd 传给 driver，持有至进程结束；锁冲突直接退出，日志保留。

ROOT 在一次新鲜 SSH 快照内核对 `runs/primary/supervisor/LAUNCH.json`、driver 的 `STARTED.json` / `PHASE.json` 与 `/proc`：当前 boot、父子 PID/start_ticks、parent PID、cmdline、cwd、12 CPU affinity 必须与 launch 一致；保存当前两进程 RSS、输出 bytes、elapsed、读取时间和 PHASE。以 `first_teacher_bank_decoded` 或 `teacher_ready` 的真实进度作为正确启动证据，并检查没有 `supervisor/FAILED.json` 或异常原生退出。若已到更后阶段，按实际已保存进度报告。通过后设一次未来苏醒并停止 agent 轮询。

终结时读取 `runs/primary/supervisor/NATIVE_EXIT.json` 与 driver `COMPLETE.json` 或 `supervisor/FAILED.json`、两个 console log；native0 与科学接受分别记账。`child_native_returncode` 原样保存；supervisor 自身 shell exit 将负信号号转换为 128+signal，资源超界即使 driver native0 也写 FAILED。选择性回收上限为 128 MiB，原始大数组保留远端。
