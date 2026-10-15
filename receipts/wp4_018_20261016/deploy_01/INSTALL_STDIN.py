from pathlib import Path
import hashlib,json,sys,tarfile
s,r=Path(sys.argv[1]),Path(sys.argv[2]); archive_name,packet_sha,cfg_sha,ops_sha,commit=sys.argv[3:]
raw=(s/'wp4_018_config.json').read_bytes(); assert hashlib.sha256(raw).hexdigest()==cfg_sha
p=json.loads(raw); assert p['run_id']==r.name and p['status']=='frozen' and len(p['source_files'])==6
manifest_name=p['saved_input_manifest']; manifest=(s/manifest_name).read_bytes(); assert hashlib.sha256(manifest).hexdigest()==p['saved_input_manifest_sha256']
ops=(s/'REMOTE_SUPERVISOR.py').read_bytes(); assert hashlib.sha256(ops).hexdigest()==ops_sha
assert hashlib.sha256((s/archive_name).read_bytes()).hexdigest()==packet_sha
payload={}
with tarfile.open(s/archive_name,'r:*') as archive:
 members=archive.getmembers(); assert sorted(m.name for m in members)==sorted(p['source_files'])
 for member in members:
  assert member.isfile(); data=archive.extractfile(member).read(); row=p['source_files'][member.name]
  assert str(r/member.name)==row['path']; assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']; payload[member.name]=data
payload.update({'wp4_018_config.json':raw,manifest_name:manifest,'REMOTE_SUPERVISOR.py':ops})
r.mkdir(exist_ok=False)
for name,data in payload.items():
 path=r/name; path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as stream:stream.write(data)
receipt=dict(run_id=r.name,config_sha256=cfg_sha,source_package_sha256=packet_sha,source_commit=commit,supervisor_sha256=ops_sha,source_files=6,scientific_started=False)
with (r/'REMOTE_INSTALL_RECEIPT.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
