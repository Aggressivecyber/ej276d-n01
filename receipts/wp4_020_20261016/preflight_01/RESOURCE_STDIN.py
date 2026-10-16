from pathlib import Path
import hashlib,json,sys
old=Path('/home/rylan/projects/ej276d-n01/wp4_017_20261015');r=old.parent/'wp4_019_20261016'
p=json.loads((r/'wp4_019_config.json').read_text())
for row in p['protection_files'].values():
 data=Path(row['path']).read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
sys.path.insert(0,str(old/'scripts'))
import wp4_017_controller
ops=wp4_017_controller.operations()
state=ops.core().runtime_preflight(r,p)
state['proposed020_root_exists']=(old.parent/'wp4_020_20261016').exists()
state['old019_native_exit']=json.loads((r/'runs/primary/supervisor/exit_receipt.json').read_text())['child_native_exit_code']
state['old019_final_checkpoint']=ops.named_record(r/'runs/primary/artifacts/T1_unpaired_seed0/final.pt')
state['UTC']=ops.core().now()
print(json.dumps(state))
