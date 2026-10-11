"""Single-process Linux resource guard; only signals its own child process group."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    tmp.replace(path)


def rss_tree(root):
    pending=[root]
    total=0
    seen=set()
    while pending:
        pid=pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            status=Path(f'/proc/{pid}/status').read_text()
            for line in status.splitlines():
                if line.startswith('VmRSS:'):
                    total+=int(line.split()[1])*1024
            children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
            pending.extend(map(int,children))
        except (FileNotFoundError,ProcessLookupError):
            pass
    return total,sorted(seen)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--rss-limit-gib',type=float,default=48.)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    command=args.command[1:] if args.command and args.command[0]=='--' else args.command
    if not command:
        raise ValueError('missing child command')
    import fcntl
    lock=(out/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (out/'console.log').open('a',buffering=1) as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=datetime.now(timezone.utc).isoformat()
        write(out/'launch.json',dict(supervisor_pid=os.getpid(),child_pid=child.pid,command=command,started_utc=started))
        peak=0
        reason=None
        while child.poll() is None:
            rss,pids=rss_tree(child.pid)
            peak=max(peak,rss)
            receipt=dict(supervisor_pid=os.getpid(),child_pid=child.pid,rss_bytes=rss,
                peak_rss_bytes=peak,child_tree=pids,utc=datetime.now(timezone.utc).isoformat())
            write(out/'resource_status.json',receipt)
            if rss>args.rss_limit_gib*1024**3:
                reason='owned_child_RSS_limit_exceeded'
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(1)
        code=child.wait()
        write(out/'exit_receipt.json',dict(exit_code=code,reason=reason,peak_rss_bytes=peak,
            started_utc=started,ended_utc=datetime.now(timezone.utc).isoformat(),
            supervisor_pid=os.getpid(),child_pid=child.pid))
        sys.exit(code if code>=0 else 1)


if __name__=='__main__':
    main()
