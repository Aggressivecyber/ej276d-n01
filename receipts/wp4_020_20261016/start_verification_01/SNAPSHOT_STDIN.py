from pathlib import Path
import hashlib,json,sys
root=Path('/home/rylan/projects/ej276d-n01/wp4_020_20261016'); cfg_sha,mf_sha,ops_sha=sys.argv[1:]
raw=(root/'wp4_020_config.json').read_bytes(); assert hashlib.sha256(raw).hexdigest()==cfg_sha
p=json.loads(raw)
for row in p['protection_files'].values():
 data=Path(row['path']).read_bytes(); assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
old=root.parent/'wp4_017_20261015';sys.path.insert(0,str(old/'scripts'))
import wp4_017_controller as inherited
ops=inherited.operations();sys.path.insert(0,str(root/'scripts'))
import run_wp4_020 as driver
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
ops.__dict__.update(RUN_ID='wp4_020_20261016',REMOTE=str(root),CFG='wp4_020_config.json',MANIFEST='WP4_020_SOURCE_MANIFEST.json',DRIVER='scripts/run_wp4_020.py',CONTROLLER='REMOTE_SUPERVISOR.py',validate_config=driver.validate_settings020)
inherited.REMOTE=str(root)
ops.STATE_NAMES=ops.STATE_NAMES+('runs/primary/artifacts/UNPAIRED_GATE020.json','runs/primary/artifacts/RESTORED_SEED0_PROOF.json',)
p,state=inherited.native_snapshot(dict(config_sha256=cfg_sha,source_manifest_sha256=mf_sha,controller_sha256=ops_sha))
state['supervisor_console_tail']=(root/'supervisor_console.log').read_text()[-8000:]
log=root/'runs/primary/console.log'
state['worker_console_tail']=log.read_text()[-16000:] if log.exists() else None
print(json.dumps(state))
