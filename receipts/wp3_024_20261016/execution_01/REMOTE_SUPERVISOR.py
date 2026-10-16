"""Fixed024 bootstrap; reuse the unchanged017 GPU/owned-tree supervisor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
parser.add_argument('--config-sha256', required=True)
args = parser.parse_args()
root = Path('/home/rylan/projects/ej276d-n01/wp3_024_20261016')
old = Path('/home/rylan/projects/ej276d-n01/wp4_017_20261015')
if Path.cwd() != root or Path(args.config).resolve() != root/'wp3_024_config.json':
    raise ValueError('024 exact owned cwd/config required')
dependencies = {
    'scripts/wp4_017_controller.py': 'eb15f3ba35844436f9b5ff3879332bd19046fcfd4f1fffe8459db2505333c38c',
    'scripts/wp2_controller.py': '3480825e94e954a46c2c633e5df6ddef514e2b7def1332bb2e7688e78ae041a3',
    'scripts/mechanism012_ops.py': '98a18eef018beebff82565da9ed5b31208cf2bff6db6bff0c492eff9bf2a7816',
    'scripts/wp1_ops.py': 'f8ddbbdb2f2f6df033e964f27bf03dcbda07055cc7378775591ef69b9caf85b5',
    'scripts/supervise_mechanism012.py': '43b6fffd076a1e6d57db67e85e433e71d3f2cda91f3a3d5fe31f0fcb430486a4',
}
for relative, expected in dependencies.items():
    if hashlib.sha256((old/relative).read_bytes()).hexdigest() != expected:
        raise ValueError('024 readonly supervision dependency changed: '+relative)
sys.path.insert(0, str(old/'scripts'))
import wp4_017_controller as inherited
ops = inherited.operations()
config_path = root/'wp3_024_config.json'
raw = config_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != args.config_sha256:
    raise ValueError('024 frozen config SHA differs')
p = json.loads(raw)
for name, record in p['source_files'].items():
    if ops.named_record(record['path']) != {key: record[key] for key in ('bytes', 'sha256')}:
        raise ValueError('024 executable source changed: '+name)
sys.path.insert(0, str(root/'scripts'))
import run_wp3_024 as driver
if 'numpy' in sys.modules or 'torch' in sys.modules:
    raise ValueError('024 supervisor imported a numerical runtime')
ops.__dict__.update(RUN_ID='wp3_024_20261016', REMOTE=str(root),
    CFG='wp3_024_config.json', MANIFEST='WP3_024_SOURCE_MANIFEST.json',
    DRIVER='scripts/run_wp3_024.py', CONTROLLER='REMOTE_SUPERVISOR.py',
    validate_config=driver.validate_settings024)
ops.validate_config(p, driver_source=(root/'scripts/run_wp3_024.py').read_bytes())
import fcntl
lock = root.parent/'.n01_scientific_dispatch.lock'
fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
preflight = ops.core().runtime_preflight(root, p)
ops.require(preflight['disk_free_bytes'] >= p['preflight']['minimum_free_disk_GiB']*1024**3,
    '024 frozen free-disk requirement not met')
for folder in ('tmp', 'cache'):
    (root/folder).mkdir(mode=0o700, exist_ok=False)
os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
    PYTHONPATH=str(root/'src'), TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'),
    CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS=str(p['CPU_threads']),
    MKL_NUM_THREADS=str(p['CPU_threads']), OPENBLAS_NUM_THREADS=str(p['CPU_threads']),
    NUMEXPR_NUM_THREADS=str(p['CPU_threads']))
pins = dict(config_sha256=args.config_sha256,
    source_manifest_sha256=ops.sha(root/'WP3_024_SOURCE_MANIFEST.json'),
    controller_sha256=ops.sha(root/'REMOTE_SUPERVISOR.py'))
identity = inherited.process_identity(os.getpid())
start = dict(schema='n01/wp3-024-start/v1', run_id='wp3_024_20261016', UTC=ops.core().now(),
    **pins, supervisor_pid=os.getpid(), supervisor_start_ticks=identity['start_ticks'],
    command=identity['command'], boot_id=preflight['boot_id'], exclusive_scientific_lock=str(lock),
    lock_held_by_supervisor=True, actual_G1_verified=False, automatic_restart=False,
    scientific_acceptance=False, preflight=preflight,
    reused_supervision_record_format='n01/wp4-017; actual run identity is wp3_024_20261016',
    readonly_supervision_dependencies={str(old/name): digest for name, digest in dependencies.items()})
ops.core().write_new(root/'DISPATCH_ATTEMPT.json', dict(start, budget_caps=p['budget_caps']))
ops.core().write_new(root/'START_RECEIPT.json', start)
raise SystemExit(ops.supervise(args.config_sha256, fd))
