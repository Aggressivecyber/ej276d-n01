from pathlib import Path
import json,subprocess
w=Path('E:/GAN/research/N01/receipts/response071_20261018/deployment_01')
c=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','rylan@100.82.231.102','python3 -'],input=(w/'REMOTE_STARTUP_REQUEST.py').read_text(),capture_output=True,text=True,timeout=75)
(w/'STARTUP_STDOUT.txt').write_text(c.stdout);(w/'STARTUP_STDERR.txt').write_text(c.stderr);assert c.returncode==0,c.stderr
v=json.loads(c.stdout);(w/'STARTUP_RAW.json').write_text(json.dumps(v,separators=(',',':'))+'\n');print(json.dumps({k:v[k] for k in ('UTC','startup_wait_seconds','processes','current_run_bytes','console_tail')}));print(json.dumps({k:x for k,x in v['files'].items() if k in ('run_01/PROGRESS.json','run_01/FAILED.json','control_01/exit_receipt.json','run_01/RESULT.json')}))
