from pathlib import Path
import json,subprocess
w=Path('E:/GAN/research/N01/receipts/response070_20261018/deployment_01');p=json.loads((w/'PACKAGE_RECEIPT.json').read_text());host='rylan@100.82.231.102'
for local,remote in [('response070_source.tar.gz',p['remote_archive']),('SOURCE_MANIFEST.json',p['remote_manifest'])]:
 c=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=12',str(w/local),host+':'+remote],capture_output=True,text=True,timeout=60)
 (w/(local+'.SCP_STDOUT.txt')).write_text(c.stdout);(w/(local+'.SCP_STDERR.txt')).write_text(c.stderr);assert c.returncode==0,c.stderr
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,'python3 -']
c=subprocess.run(cmd,input=(w/'REMOTE_INSTALL_REQUEST.py').read_text(),capture_output=True,text=True,timeout=180)
(w/'INSTALL_STDOUT.txt').write_text(c.stdout);(w/'INSTALL_STDERR.txt').write_text(c.stderr);assert c.returncode==0,c.stderr
v=json.loads(c.stdout);(w/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
