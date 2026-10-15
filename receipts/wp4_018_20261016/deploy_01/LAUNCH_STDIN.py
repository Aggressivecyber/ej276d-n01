from pathlib import Path
import hashlib,json,subprocess,sys
r=Path(sys.argv[1]); cfg_sha,ops_sha=sys.argv[2:]
assert hashlib.sha256((r/'wp4_018_config.json').read_bytes()).hexdigest()==cfg_sha
assert hashlib.sha256((r/'REMOTE_SUPERVISOR.py').read_bytes()).hexdigest()==ops_sha
command=[sys.executable,'-B',str(r/'REMOTE_SUPERVISOR.py'),'--config',str(r/'wp4_018_config.json'),'--config-sha256',cfg_sha]
with (r/'supervisor_console.log').open('xb') as log:child=subprocess.Popen(command,cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(supervisor_pid=child.pid,command=command,config_sha256=cfg_sha)))
