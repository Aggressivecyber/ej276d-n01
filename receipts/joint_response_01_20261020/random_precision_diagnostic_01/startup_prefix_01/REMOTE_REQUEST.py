from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/rylan/projects/ej276d-n01/joint_response01_random_precision_diagnostic_20261020')
names=['run_01/seed0_rep-1_REPLAY.json','run_01/seed0_rep-1_REPLAY.npz','run_01/seed0_rep0_IID.json','run_01/seed0_rep0_IID.npz','run_01/seed0_rep0_BANK.npz']
d=json.loads((root/names[2]).read_bytes());assert d['FD_columns_completed']==54 and len(d['checks'])==2
records={}
for name in names:
 p=root/name;b=p.read_bytes();records[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),mtime_ns=p.stat().st_mtime_ns)
archive=root/'startup_prefix_01.tar';assert not archive.exists()
with tarfile.open(archive,'w') as t:
 for name in names:t.add(root/name,arcname=name,recursive=False)
for name,record in records.items():
 p=root/name;assert p.stat().st_mtime_ns==record['mtime_ns'] and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
print(json.dumps(dict(outputs=records,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
