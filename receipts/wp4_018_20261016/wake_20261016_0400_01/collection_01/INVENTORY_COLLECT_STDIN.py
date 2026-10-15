from pathlib import Path
import hashlib,json,tarfile
r=Path('/home/rylan/projects/ej276d-n01/wp4_018_20261016');a=r/'runs/primary/artifacts';s=r/'runs/primary/supervisor'
exit=json.loads((s/'NATIVE_EXIT.json').read_text());complete=json.loads((a/'COMPLETE.json').read_text());assert exit['child_native_returncode']==0 and not exit['resource_violations'] and exit['reason'] is None and not (s/'FAILED.json').exists()
assert complete['config_sha256']==exit['config_sha256']=='3eb6d180e63a7cbd4560da5cb4b2e99e10054ff5504aef1b96212c58cc8acc8a'
assert complete['distance_counts']=={'19':870908928,'257':303805440}
assert not Path('/proc/29239').exists() and not Path('/proc/29246').exists()
recorded=json.loads((a/'OUTPUT_INVENTORY.json').read_text());whole={}
for p in sorted(r.rglob('*')):
 if not p.is_file():continue
 raw=p.read_bytes();entry={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};whole[p.relative_to(r).as_posix()]=entry
 rel=p.relative_to(a).as_posix() if p.is_relative_to(a) else None
 if rel in recorded:assert recorded[rel]==entry,rel
assert all('runs/primary/artifacts/'+n in whole for n in recorded)
selected={n:v for n,v in whole.items() if not n.endswith('_OBSERVABLE.npz')}
omitted={n:v for n,v in whole.items() if n not in selected}
assert len(omitted)==12 and sum(v['bytes'] for v in selected.values())<=128*1024**2
archive=r.parent/'wp4_018_selected_20261016_0400_01.tar';assert not archive.exists()
with tarfile.open(archive,'x') as tar:
 for n in selected:tar.add(r/n,arcname=n,recursive=False)
assert archive.stat().st_size<=128*1024**2
receipt={'whole':whole,'selected':selected,'omitted':omitted,'selected_archive_path':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'original_output_inventory_verified':True,'raw_O19_arrays_retained_remote':True,'native_exit':exit,'stage1_gate':json.loads((a/'STAGE1_GATE018.json').read_text()),'complete':complete}
print(json.dumps(receipt))
