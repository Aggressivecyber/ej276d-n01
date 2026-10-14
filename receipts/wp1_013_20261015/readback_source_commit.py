"""Compare frozen executable/control bytes with their saved Git commit blobs."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
cfg = json.loads((root / 'wp1_013_config.json').read_text('utf-8'))
assert cfg['status'] == 'frozen'
expected = {name: record['sha256'] for name, record in cfg['source_files'].items()}
for name in ('wp1_013_config.json', 'WP1_SOURCE_MANIFEST.json'):
    expected[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
for relative, want in expected.items():
    local = (root / relative).read_bytes()
    committed = subprocess.check_output(['git', 'show', head + ':' + relative], cwd=root)
    assert hashlib.sha256(local).hexdigest() == hashlib.sha256(committed).hexdigest() == want, relative
result = {'schema': 'n01/wp1-source-commit-readback/v1', 'run_id': 'wp1_013_20261015',
    'UTC': datetime.now(timezone.utc).isoformat(), 'commit': head, 'frozen_source_and_control_files': len(expected),
    'exact_working_and_Git_blob_sha256': expected, 'all_passed': True, 'public_push_performed': False,
    'private_waveform_weight_or_package_committed': False, 'scientific_execution': 0}
out = root / 'receipts/wp1_013_20261015/SOURCE_COMMIT_READBACK.json'
with out.open('xb') as handle:
    handle.write((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps({'commit': head, 'files_verified': len(expected), 'all_passed': True}))
