from pathlib import Path
from datetime import datetime,timezone
import json,os,signal,time
p=Path('/proc/215696');before=(p/'stat').read_text().rsplit(')',1)[1].split();assert before[19]=='5119820';argv=(p/'cmdline').read_bytes().split(b'\0');assert argv[:3]==[b'/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python',b'-B',b'-'];os.kill(215696,signal.SIGTERM);deadline=time.monotonic()+20
while p.exists() and time.monotonic()<deadline:time.sleep(.2)
if p.exists():os.kill(215696,signal.SIGKILL)
print(json.dumps(dict(status='OWN_DEPLOYMENT_STOP_SIGNAL_SENT',UTC=datetime.now(timezone.utc).isoformat(),pid=215696,start_ticks='5119820',before_state=before[0],process_still_present=p.exists(),after_state=(p/'stat').read_text().rsplit(')',1)[1].split()[0] if p.exists() else None,original_scientific_job_restarted=False,source_payloads_deleted=False)))
