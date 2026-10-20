from pathlib import Path
import datetime, hashlib, json, tarfile

root = Path('/home/rylan/projects/ej276d-n01/joint_response01_curvature_update_diagnostic_20261020')
manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text())
assert manifest['config_sha256'] == 'f91086896da8c0f6fe14fd946551acd7eda5c9cbf91b96d2dad0c5757270d24e'
for name,item in manifest['files'].items():
    raw = (root/name).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], name
ex = json.loads((root/'control_01/exit_receipt.json').read_text())
processes = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        argv = [a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(bytes([0])) if a]
    except (FileNotFoundError, ProcessLookupError, PermissionError): continue
    if any(str(root) in a for a in argv): processes.append(dict(pid=int(proc.name), argv=argv))
assert not processes, processes
outputs = {}
for sub in ('run_01','control_01'):
    for path in (root/sub).rglob('*'):
        if path.is_file():
            raw = path.read_bytes()
            outputs[str(path.relative_to(root))] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), mtime_ns=path.stat().st_mtime_ns)
assert sum(v['bytes'] for v in outputs.values()) < 64*1024**2
archive = root/'curvature_update_diagnostic_terminal_01.tar'
assert not archive.exists()
with tarfile.open(archive, 'w') as tar:
    for name in sorted(outputs): tar.add(root/name, arcname=name, recursive=False)
for name,item in outputs.items():
    path=root/name; raw=path.read_bytes()
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'] and path.stat().st_mtime_ns==item['mtime_ns'], name
print(json.dumps(dict(status='PASS_TERMINAL_COLLECTION', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_files_verified=len(manifest['files']), outputs=outputs, processes=processes,
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), native_exit=ex,
    archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    complete_present=(root/'run_01/COMPLETE.json').exists(), failed_present=(root/'run_01/FAILED.json').exists(),
    result_present=(root/'run_01/RESULT.json').exists())))
