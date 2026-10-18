import datetime,hashlib,json,os,tarfile
from pathlib import Path
root=Path('/home/rylan/projects/ej276d-n01/response070_split_area_crossing_level_warp_20261018')
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
identities={}
for pid,expected in ((260491, '7313248'), (260492, '7313248'), (260498, '7313250'), (260500, '7313250')):
 proc=Path('/proc')/str(pid)
 if not proc.exists():identities[str(pid)]=dict(present=False,expected_start_ticks=expected);continue
 try:
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
  argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(bytes([0])) if v]
  identities[str(pid)]=dict(present=True,start_ticks=stat[19],expected_start_ticks=expected,
   same_start_ticks=stat[19]==expected,state=stat[0],argv=argv)
 except (FileNotFoundError,ProcessLookupError):identities[str(pid)]=dict(present=False,exited_during_snapshot=True)
files={}
for rel in ('control_01/exit_receipt.json','control_01/resource_status.json','run_01/RESULT.json',
 'run_01/COMPLETE.json','run_01/FAILED.json','run_01/PROGRESS.json','run_01/ENDPOINT_FROZEN.json'):
 path=root/rel
 if path.exists():files[rel]=json.loads(path.read_text())
record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=boot,
 expected_boot_id='b6923bf3-a173-4b1a-a175-f746178ee99c',identities=identities,files=files)
terminal=(root/'control_01/exit_receipt.json').exists()
if terminal:
 if any(z.get('present') and z.get('same_start_ticks') and z.get('state')!='Z' for z in identities.values()):raise RuntimeError('070 terminal receipt with live expected process')
 source=json.loads((root/'SOURCE_MANIFEST.json').read_text())
 for n,m in source['files'].items():
  b=(root/n).read_bytes()
  if len(b)!=m['bytes'] or hashlib.sha256(b).hexdigest()!=m['sha256']:raise ValueError('070 terminal source mismatch '+n)
 record['terminal_source_verification']=dict(source_files=len(source['files']),all_hashes_verified=True,config_sha256=source['config_sha256'])
 paths=sorted([p for sub in ('run_01','control_01') for p in (root/sub).rglob('*') if p.is_file()]+
              [root/name for name in ('DISPATCH_RECEIPT.json','DEPLOYMENT_VERIFICATION.json','SOURCE_MANIFEST.json')])
 if any(p.is_symlink() for p in paths):raise ValueError('070 output symlink outside collection contract')
 size=sum(p.stat().st_size for p in paths)
 if size>64*1024**2:raise ValueError('070 output exceeds bounded collection cap')
 manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
 dest=root/'collection_20261018_2043_01';dest.mkdir()
 mp=dest/'MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
 archive=dest/'response070_outputs.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for path in paths:tf.add(path,arcname=str(path.relative_to(root)),recursive=False)
 record['collection']=dict(manifest=manifest,manifest_path=str(mp),archive_path=str(archive),
  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,
  uncompressed_bytes=size,file_count=len(paths),no_checkpoints_or_raw_dataset_decoded=True)
record['console_tail']=(root/'control_01/console.log').read_text(errors='replace')[-4000:]
print(json.dumps(record))
