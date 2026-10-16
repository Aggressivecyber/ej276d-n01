from pathlib import Path
import hashlib,json,os,subprocess,sys

root=Path('/home/rylan/projects/ej276d-n01/wp3_025_20261016')
archive_path,cfg_sha,mf_sha,ops_sha,archive_sha=sys.argv[1:]
old=root.parent/'wp4_017_20261015'
dependencies={
 'scripts/wp4_017_controller.py':'eb15f3ba35844436f9b5ff3879332bd19046fcfd4f1fffe8459db2505333c38c',
 'scripts/wp2_controller.py':'3480825e94e954a46c2c633e5df6ddef514e2b7def1332bb2e7688e78ae041a3',
 'scripts/mechanism012_ops.py':'98a18eef018beebff82565da9ed5b31208cf2bff6db6bff0c492eff9bf2a7816',
 'scripts/wp1_ops.py':'f8ddbbdb2f2f6df033e964f27bf03dcbda07055cc7378775591ef69b9caf85b5',
 'scripts/supervise_mechanism012.py':'43b6fffd076a1e6d57db67e85e433e71d3f2cda91f3a3d5fe31f0fcb430486a4'}
for name,digest in dependencies.items():
 assert hashlib.sha256((old/name).read_bytes()).hexdigest()==digest
sys.path.insert(0,str(old/'scripts'))
import wp4_017_controller as inherited
ops=inherited.operations()
assert ops.sha(archive_path)==archive_sha
inventory=ops.index(archive_path,16*1024**2)
ops.core().exclusive_extract(archive_path,root,inventory,16*1024**2)
assert ops.sha(root/'wp3_025_config.json')==cfg_sha
assert ops.sha(root/'WP3_025_SOURCE_MANIFEST.json')==mf_sha
assert ops.sha(root/'REMOTE_SUPERVISOR.py')==ops_sha
sys.path.insert(0,str(root/'scripts'))
import run_wp3_025 as driver
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
p,inputs,checked,stats,tolerance=driver.gate025(root/'wp3_025_config.json',cfg_sha,root/'runs/primary/artifacts')
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
command=[sys.executable,'-B',str(root/'REMOTE_SUPERVISOR.py'),'--config',str(root/'wp3_025_config.json'),'--config-sha256',cfg_sha]
with (root/'supervisor_console.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,
  stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1'))
print(json.dumps(dict(run_id=p['run_id'],root=str(root),supervisor_pid=child.pid,
 command=command,archive_sha256=archive_sha,archive_members=len(inventory),
 immutable_paths_verified=len(checked),runtime_gate_errors=[],
 source_science_files=len(driver.SCIENCE_SOURCES025),scientific_completion_not_yet_verified=True)))
