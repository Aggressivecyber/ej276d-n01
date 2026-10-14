"""Independent fake process/resource tests; never spawn/signal a real process."""
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path('E:/GAN/research/N01')
OWN = Path(__file__).resolve().parent
OUTPUT = OWN/'supervisor_mock_02'
OUTPUT.mkdir(exist_ok=False)
SOURCE = ROOT/'scripts/supervise_stability011.py'
SOURCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('independent_supervisor011', SOURCE)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
policy = json.loads((ROOT/'stability011_policy.json').read_text(encoding='utf-8'))
limits = s.validate_limits(policy)
passed = []


def run_test(name, test):
    test()
    passed.append(dict(name=name, passed=True))


def recursive_owned_RSS_only():
    ps = 'PID PPID RSS\n100 1 1000\n101 100 2000\n102 101 3000\n200 1 8000000\n201 200 2000000\n'
    with patch.object(s.subprocess, 'check_output', return_value=ps) as call:
        pids, size = s.process_tree(100)
    assert pids == {100,101,102} and size == 6000*1024
    assert call.call_args.args[0] == ['ps','-eo','pid,ppid,rss']


def GPU_ownership_and_zero_GPU_budget():
    table = '100, 0\n102, 1\n200, 6000\n'
    with patch.object(s.subprocess, 'check_output', return_value=table):
        assert s.gpu_bytes({100,102}) == 1024**2
        assert s.gpu_bytes({100}) == 0
    assert s.limit_reason(limits, rss=1, gpu=1024**2, size=1, elapsed=1) == 'unexpected_GPU_computation'


def strict_resource_limits_and_interrupt():
    boundary = dict(rss=8*1024**3, gpu=0, size=256*1024**2, elapsed=7200)
    assert s.limit_reason(limits, **boundary) is None
    for key, label in [('rss','RSS_limit'),('size','output_limit'),('elapsed','wall_limit')]:
        values = dict(boundary); values[key] += 1
        assert s.limit_reason(limits, **values) == label
    assert s.limit_reason(limits, **boundary, interrupted=2) == 'user_signal'


def signal_only_owned_group_and_never_completed_child():
    class Child:
        pid = 987651
        waits = 0
        def poll(self): return None
        def wait(self, timeout=None):
            self.waits += 1
            if timeout is not None:
                raise s.subprocess.TimeoutExpired('artificial-child', timeout)
            return -9
    child = Child()
    with patch.object(s.os, 'killpg', create=True) as kill, patch.object(s.signal, 'SIGKILL', 9, create=True):
        s.stop_owned(child, 20)
        assert [(c.args[0],c.args[1]) for c in kill.call_args_list] == [(987651,s.signal.SIGTERM),(987651,9)]
        assert child.waits == 2
        class Completed:
            pid = 987652
            def poll(self): return 0
            def wait(self, timeout=None): raise AssertionError('Completed child re-waited')
        s.stop_owned(Completed(), 20)
        assert kill.call_count == 2


def partial_progress_and_output_count_readback():
    output = OUTPUT/'fake_output'; output.mkdir()
    no_status = s.last_progress(output)
    assert no_status['completed_replicates'] is None and no_status['in_flight_actual_completion'] == 'unknown'
    status = dict(bootstrap_replicates_completed=1,bootstrap_replicates_attempted=3,
                  bootstrap_replicates_computed_successfully=2,phase='bootstrap',artificial_only=True)
    (output/'status.json').write_text(json.dumps(status),encoding='utf-8')
    progress = s.last_progress(output)
    assert progress['completed_replicates'] == 1 and progress['attempted_replicates_persisted_lower_bound'] == 3
    assert progress['actual_unpersisted_attempts'] == 'unknown outside durable child receipts'
    assert progress['last_persisted_status'] == status
    assert 'only persisted completed' in progress['in_flight_actual_completion']
    (output/'dummy.bin').write_bytes(b'x'*17)
    assert s.output_bytes(output) == (output/'status.json').stat().st_size+17


try:
    for name, test in [('recursive_owned_process_tree_RSS',recursive_owned_RSS_only),
            ('owned_GPU_only_and_forbidden_GPU_guard',GPU_ownership_and_zero_GPU_budget),
            ('cap_boundary_and_interrupt_reason',strict_resource_limits_and_interrupt),
            ('TERM_KILL_owned_group_completed_child_guard',signal_only_owned_group_and_never_completed_child),
            ('partial_persisted_progress_and_output_count',partial_progress_and_output_count_readback)]:
        run_test(name,test)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    value = dict(schema='n01/stability011-independent-supervisor-mocks/v1',UTC=datetime.now(timezone.utc).isoformat(),
        status='passed',source_sha256=SOURCE_SHA,checks=passed,passed_checks=len(passed),
        process_enumerations=0,real_signal_calls=0,real_child_launches=0,SSH_or_SCP_calls=0,
        model_instances=0,scientific_bootstrap_replicates=0,scientific_array_decodes=0,
        mock_only=True,platform='Windows fake Linux process/signal API',
        exclusive_flock_inheritance='static reviewed; native POSIX kernel enforcement is a later root-launch proof')
    with (OUTPUT/'ENGINEERING_RECEIPT.json').open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(value))
except BaseException as error:
    with (OUTPUT/'FAILURE_RECEIPT.json').open('x',encoding='utf-8') as f:
        json.dump(dict(error_type=type(error).__name__,error=str(error),passed=passed,mock_only=True),f,indent=2)
    raise
