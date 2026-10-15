from pathlib import Path
import hashlib,json,subprocess,sys
r=Path(sys.argv[1]); cfg_sha,mf_sha,ops_sha=sys.argv[2:]
for name,expected in [('wp4_019_config.json',cfg_sha),('WP4_019_SOURCE_MANIFEST.json',mf_sha),('REMOTE_SUPERVISOR.py',ops_sha)]:
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected
command=[sys.executable,'-B',str(r/'REMOTE_SUPERVISOR.py'),'--config',str(r/'wp4_019_config.json'),'--config-sha256',cfg_sha]
with (r/'supervisor_console.log').open('xb') as log:child=subprocess.Popen(command,cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(supervisor_pid=child.pid,command=command,config_sha256=cfg_sha)))
