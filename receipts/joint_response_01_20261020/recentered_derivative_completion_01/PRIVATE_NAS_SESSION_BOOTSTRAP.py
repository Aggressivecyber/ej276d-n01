from datetime import datetime, timezone
from pathlib import Path
import json
import os
import pwd
import subprocess
import time

session_started = time.monotonic()
global_namespace = 'mnt:[4026531832]'
private_namespace = str(Path('/proc/self/ns/mnt').readlink())
assert private_namespace != global_namespace
session_mount_command = ['mount', '--no-canonicalize', '-t', 'cifs',
    '//192.168.31.38/我的文档-1430134245', '/mnt/MiNas', '-o',
    'guest,sec=none,vers=3.1.1,nosharesock,uid=1000,gid=1000,forceuid,forcegid,iocharset=utf8,file_mode=0600,dir_mode=0700,nosuid,nodev']
subprocess.run(session_mount_command, capture_output=True, check=True, timeout=30)
private_mount = json.loads(subprocess.check_output(['findmnt', '--json', '--target', '/mnt/MiNas', '-o', 'SOURCE,FSTYPE,PROPAGATION,OPTIONS'], text=True))
assert all(item['propagation'] == 'private' for item in private_mount['filesystems'])
account = pwd.getpwnam('rylan')
os.initgroups(account.pw_name, account.pw_gid)
os.setgid(account.pw_gid)
os.setuid(account.pw_uid)
print(json.dumps(dict(stage='N01_PRIVATE_NAS_SESSION_READY', UTC=datetime.now(timezone.utc).isoformat(),
    global_mount_namespace=global_namespace, private_mount_namespace=private_namespace,
    mount=private_mount, UID=os.getuid(), elapsed_seconds=time.monotonic()-session_started,
    persistent_global_mount_modified=False, data_migration_performed=False)), flush=True)
