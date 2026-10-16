from pathlib import Path
import hashlib,json,sys,tarfile
s,r=Path(sys.argv[1]),Path(sys.argv[2]); packet_sha,cfg_sha,mf_sha,ops_sha,commit=sys.argv[3:]
raw=(s/'wp4_023_config.json').read_bytes(); assert hashlib.sha256(raw).hexdigest()==cfg_sha
p=json.loads(raw); assert p['run_id']==r.name and p['status']=='frozen'
mraw=(s/'WP4_023_SOURCE_MANIFEST.json').read_bytes(); assert hashlib.sha256(mraw).hexdigest()==mf_sha
m=json.loads(mraw); assert m['config_sha256']==cfg_sha and set(m['files'])==set(p['source_files'])
archive=s/'wp4_023_source.tar.gz'; assert hashlib.sha256(archive.read_bytes()).hexdigest()==packet_sha
payload={}
with tarfile.open(archive,'r:*') as archive:
 members=archive.getmembers(); assert len(members)==len(m['files']) and sorted(x.name for x in members)==sorted(p['source_files'])
 for member in members:
  assert member.isfile(); name=member.name; target=r/name; assert target.is_relative_to(r)
  data=archive.extractfile(member).read(); row=p['source_files'][name]
  assert str(target)==row['path'] and m['files'][name]=={k:row[k] for k in ('bytes','sha256')}
  assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']; payload[name]=data
assert hashlib.sha256(payload['REMOTE_SUPERVISOR.py']).hexdigest()==ops_sha
assert hashlib.sha256(payload[p['saved_input_manifest']]).hexdigest()==p['saved_input_manifest_sha256']
payload.update({'wp4_023_config.json':raw,'WP4_023_SOURCE_MANIFEST.json':mraw})
r.mkdir(exist_ok=False)
for name,data in payload.items():
 path=r/name; path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as stream:stream.write(data)
receipt=dict(run_id=r.name,config_sha256=cfg_sha,source_manifest_sha256=mf_sha,source_package_sha256=packet_sha,source_commit=commit,supervisor_sha256=ops_sha,source_files=len(m['files']),scientific_started=False)
with (r/'REMOTE_INSTALL_RECEIPT.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
