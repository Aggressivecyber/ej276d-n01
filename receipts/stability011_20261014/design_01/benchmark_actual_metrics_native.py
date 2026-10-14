"""Send actual NumPy CDF helper plus a synthetic benchmark over SSH stdin."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

own = Path(__file__).resolve().parent
source_path = own.parents[2]/'scripts/stability011_metrics.py'
source = source_path.read_text(encoding='utf8')
prefix = "import os,sys,types\nfor k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'\naff=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,aff[:12])\nm=types.ModuleType('synthetic_stability011_metrics');sys.modules[m.__name__]=m\n"
payload = prefix+'exec(compile('+repr(source)+",'<verified actual metrics source>','exec'),m.__dict__)\n"+'''
import numpy as np,time,json,resource
from concurrent.futures import ThreadPoolExecutor
rng=np.random.Generator(np.random.PCG64(2026101457))
begin=time.perf_counter()
c=m.CDFCache.create(np.round(rng.normal(size=(2048,1024)),3),np.round(rng.normal(size=(2048,1024)),3))
prep=time.perf_counter()-begin
parent=np.repeat(np.arange(512),4);groups=np.arange(2048)%419
fs=rng.integers(0,512,size=(24,512));rs=rng.integers(0,419,size=(24,419))
fc=np.stack([np.bincount(v,minlength=512) for v in fs]);rc=np.stack([np.bincount(v,minlength=419) for v in rs])
def work(i):return c.distance(fc[i][parent],rc[i][groups],column_chunk=256)
begin=time.perf_counter()
with ThreadPoolExecutor(max_workers=12) as p:values=list(p.map(work,range(24)))
wall=time.perf_counter()-begin
print(json.dumps(dict(artificial_only=True,scientific_bootstrap_replicates=0,private_array_decodes=0,checkpoint_decodes=0,NN_forwards=0,
optimizer_updates=0,CPU_affinity=sorted(os.sched_getaffinity(0)),threads=12,NumPy_version=np.__version__,synthetic_rows=[2048,2048],columns=1024,
cache_prepare_seconds=prep,cached_reductions=24,reducer_wall_seconds=wall,cache_bytes=c.bytes,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
linear_cached_reducer_estimate_seconds=wall/24*3000*8794/1024,checksum=float(sum(values)),
estimate_scope='Synthetic actual helper, twelve threads, 2048 rows/domain, 1024 columns; excludes true bank decode/projection/full cache sorting/hash/fsync/QA; 6 comparisons x500 only.')))
'''
command = ['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','rylan@100.82.231.102',
    'env PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -']
result = subprocess.run(command,input=payload.encode('utf8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),artificial_only=True,native_ssh_exit_code=result.returncode,
    source_sha256=hashlib.sha256(source.encode('utf8')).hexdigest(),stdin_sha256=hashlib.sha256(payload.encode('utf8')).hexdigest(),
    benchmark=json.loads(result.stdout.decode('utf8')) if result.returncode==0 else None,
    stderr=result.stderr.decode('utf8',errors='replace'),remote_persistent_output=False)
with (own/'ACTUAL_METRICS_NATIVE_BENCHMARK.json').open('x',encoding='utf8') as stream:
    json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(receipt,allow_nan=False))
raise SystemExit(result.returncode)
