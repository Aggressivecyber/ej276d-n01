"""Promote controller-produced control bytes after their complete freeze passes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

root = Path('E:/GAN/research/N01')
parser = argparse.ArgumentParser()
parser.add_argument('--freeze', required=True, type=Path)
args = parser.parse_args()
freeze = args.freeze.resolve(strict=True)
assert freeze.is_relative_to(root)
receipt = json.loads((freeze / 'FREEZE_RECEIPT.json').read_text('utf-8'))
assert receipt['schema'] == 'n01/wp1-freeze/v1' and receipt['run_id'] == 'wp1_013_20261015'
assert receipt['all_local_original_files_unchanged'] and not receipt['scientific_started']
cfg_name, mf_name = 'wp1_013_config.json', 'WP1_SOURCE_MANIFEST.json'
pins = {cfg_name: receipt['config_sha256'], mf_name: receipt['source_manifest_sha256']}
for name, want in pins.items():
    assert hashlib.sha256((freeze / name).read_bytes()).hexdigest() == want
cfg = json.loads((freeze / cfg_name).read_text('utf-8'))
assert cfg['status'] == 'frozen' and cfg['source_files'] and cfg['scientific_training_updates'] == 0
history = root / 'receipts/wp1_013_20261015/draft_versions/05_before_adoption'
history.mkdir(parents=True, exist_ok=False)
assert json.loads((root / cfg_name).read_text('utf-8'))['status'] == 'draft_registered'
shutil.copyfile(root / cfg_name, history / cfg_name)
assert not (root / mf_name).exists()
shutil.copyfile(freeze / cfg_name, root / cfg_name)
shutil.copyfile(freeze / mf_name, root / mf_name)
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == want for name, want in pins.items())
out = root / 'receipts/wp1_013_20261015/FROZEN_CONTROL_ADOPTION.json'
with out.open('xb') as handle:
    handle.write((json.dumps({'schema': 'n01/wp1-frozen-controls-adoption/v1', 'run_id': receipt['run_id'],
        'exact_frozen_controls': pins, 'source_files': len(cfg['source_files']),
        'old_draft_preserved': str(history / cfg_name), 'private_inputs_not_copied_or_decoded': True,
        'scientific_training_updates': 0}, indent=2) + '\n').encode('utf-8'))
print(json.dumps({'stage': 'controller_bytes_adopted', 'controls': pins, 'source_files': len(cfg['source_files'])}))
