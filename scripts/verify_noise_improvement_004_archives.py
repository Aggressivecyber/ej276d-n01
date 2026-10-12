"""Reopen both archives and bind every member to its original bytes."""
from pathlib import Path
import datetime as dt
import hashlib
import io
import json
import zipfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ID = 'noise_improvement_004_20261012'
RUN = ROOT / 'runs' / ID
DEST = RUN / 'delivery'
REC = ROOT / 'receipts' / ID


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    inventory = json.loads((DEST / 'PUBLIC_INVENTORY.json').read_text(encoding='utf8'))
    evidence = []
    for label in ['private-full', 'public-derived']:
        path = DEST / ('n01-noise-improvement-004-' + label + '.zip')
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            assert len(names) == len(set(names))
            assert all(not n.startswith('/') and '..' not in Path(n).parts for n in names)
            if label == 'public-derived':
                lookup = {x['path']: x for x in inventory['files']}
                assert set(names) == set(lookup) | {'PUBLIC_INVENTORY.json'}
                assert not any('baseline_populations' in n or 'TRAIN_PRIVATE' in n or 'PARENT_IDENTITIES' in n for n in names)
                assert archive.read('PUBLIC_INVENTORY.json') == (DEST / 'PUBLIC_INVENTORY.json').read_bytes()
                for name, item in lookup.items():
                    data = archive.read(name)
                    assert len(data) == item['bytes'] and sha(data) == item['sha256'], name
                    if name.endswith('predictions.npz'):
                        with np.load(io.BytesIO(data), allow_pickle=False) as predictions:
                            assert set(predictions.files) == {'labels', 'scores', 'groups0', 'groups1', 'loss'}, (name, predictions.files)
                            assert predictions['labels'].shape == predictions['scores'].shape == (512,)
                    if name.endswith('.pt'):
                        state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
                        assert isinstance(state, dict)
                        forbidden = {'train_waveforms', 'real_waveforms', 'validation_waveforms', 'train_x', 'test_x', 'real_train', 'real_validation'}
                        assert not (set(state) & forbidden), name
            else:
                roots = {'runs': RUN / 'runs', 'report': ROOT / 'reports' / ID,
                         'receipts': REC, 'visual_QA': RUN}
                for name in names:
                    parts = name.split('/', 1)
                    if parts[0] in roots:
                        source = roots[parts[0]] / parts[1]
                    elif name == 'EXECUTION_SOURCE.tar.gz':
                        source = RUN / name
                    elif name == 'n01-noise-improvement-004-source.bundle':
                        source = DEST / name
                    else:
                        raise AssertionError(name)
                    assert sha(archive.read(name)) == sha(source.read_bytes()), name
            evidence.append(dict(archive=path.name, members_verified=len(names),
                                 all_member_sha256_match=True, sha256=sha(path.read_bytes())))
    result = dict(status='all_archive_members_verified', UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        archives=evidence, public_waveform_population_npz_excluded=True,
        public_prediction_npz_schema_verified=True, new_scientific_updates=0)
    (REC / 'ARCHIVE_MEMBER_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
