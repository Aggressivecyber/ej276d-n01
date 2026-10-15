"""Preserve the one failed bootstrap attempt without any further native calls."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
stage=ROOT/'receipts/wp3_016_20261015'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pre=read(HERE/'LOCAL_PRELAUNCH_VALIDATION.json')
receipt_path=stage/'deploy_native_01/prepare_NATIVE_RECEIPT.json';receipt=read(receipt_path)
assert receipt['native_ssh_exit_code']==1 and '016 exact reused014 operations unavailable' in receipt['stderr']
assert not (stage/'start_native_01').exists() and not (stage/'start_verification_01').exists()
record=dict(schema='n01/wp3-016-native-start-attempt/v1',UTC=datetime.now(timezone.utc).isoformat(),
    status='DEPLOY_BOOTSTRAP_FAILED_BEFORE_INSTALL_OR_START',passed=False,source_commit=pre['source_commit'],pins=pre['pins'],
    prelaunch_validation_sha256=sha(HERE/'LOCAL_PRELAUNCH_VALIDATION.json'),
    deploy_receipt=dict(path=receipt_path.relative_to(ROOT).as_posix(),sha256=sha(receipt_path)),
    actions=dict(deploy_CLI_calls=1,SSH_bootstrap_prepare_calls=1,SCP_source_transfers=0,install_calls=0,start_calls=0,
        first_event_wait_calls=0,verify_start_snapshot_calls=0,collect_calls=0,automatic_retry=False),
    native_result=dict(SSH_exit_code=1,stdout_empty=True,exception='ValueError: 016 exact reused014 operations unavailable',
        failure_stage='controller operations loader before dispatch to native prepare/install/start',
        reason='stdin bootstrap SELF_ROOT points to the new016 directory; operations omitted the mature015 fallback to the existing pinned014 controller'),
    actual_start_UTC=None,first_update_UTC=None,same_snapshot_UTC=None,supervisor_pid=None,child_pid=None,start_ticks=None,
    observed_scientific_counters=None,attempted_scientific_counters=None,unknown_inflight=None,
    scientific_launch_performed=False,scientific_cost_from_this_failed_bootstrap=0,
    scope='No scientific launch action was reached. No fresh server status snapshot was requested after the bootstrap exception.',
    resource_health='not sampled; start preflight not reached',post_failure_remote_activity='none',
    metadata_or_automation_modified=False,
    next_action='Root to authorize a minimal bootstrap fallback source fix, refreeze/review/commit/back up new exact pins, then decide a new explicit deployment attempt.',
    weights_promoted=False,scientific_acceptance=False)
with (HERE/'NATIVE_START_SUMMARY.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
md=f'''# 016 native部署尝试

唯一一次deploy在bootstrap准备阶段失败，SSH退出1，未进入install或start；首事件等待和verify-start均未执行。

失败原因：stdin bootstrap使用尚未创建的016目录作为SELF_ROOT，新控制器operations仅查该目录，遗漏015已有的旧014固定SHA源码fallback。异常发生在native操作分派之前，本次bootstrap没有启动科学作业。

原失败回执：`deploy_native_01/prepare_NATIVE_RECEIPT.json`，SHA256 `{sha(receipt_path)}`。

源码提交：`{pre['source_commit']}`。配置、manifest、package和controller的原始pins保存在同级NATIVE_START_SUMMARY.json。

动作计数：deploy1、bootstrap SSH1、source SCP0、install0、start0、wait0、verify0；自动重试0。未追加任何SSH、进度读取、科学计算或状态快照。当前服务器资源及其它远端状态未经本次重新观察。

下一步由root处理最小fallback修复、重新冻包/独审/commit/D备份及新尝试授权。原pins和失败证据保留。
'''
with (HERE/'NATIVE_START_SUMMARY.md').open('x',encoding='utf-8') as f:f.write(md)
print(json.dumps(dict(status=record['status'],summary=str(HERE/'NATIVE_START_SUMMARY.json'),sha256=sha(HERE/'NATIVE_START_SUMMARY.json'))))
