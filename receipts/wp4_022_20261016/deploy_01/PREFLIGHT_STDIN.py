from pathlib import Path
import hashlib,json,subprocess,sys
old=Path('/home/rylan/projects/ej276d-n01/wp4_017_20261015');base=old.parent
p=json.loads(sys.argv[1])
for row in p['protection_files'].values():
 raw=Path(row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
sys.path.insert(0,str(old/'scripts'))
import wp4_017_controller as inherited
ops=inherited.operations();preflight=ops.core().runtime_preflight(base,p)
assert preflight['disk_free_bytes']>=40*1024**3
commands=subprocess.check_output(['ps','-eo','pid,args'],text=True,timeout=10).splitlines()
project=[line for line in commands if '/ej276d-n01/' in line and any(marker in line for marker in ('run_wp','REMOTE_SUPERVISOR.py','supervise_','n01.'))]
assert project==[],project
print(json.dumps(dict(UTC=ops.core().now(),run_id='wp4_022_20261016',preflight=preflight,project_science_commands=project,numerical_imports=False,scientific_started=False)))
