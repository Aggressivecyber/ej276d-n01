"""Hash immutable source manifests and bound recovery references; decode no data."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from metadata_audit import ROOT, OUT, sha


def main():
    manifests = {}
    for name, expected_count in [('WINDOW008_SOURCE_MANIFEST.json', 68),
                                 ('WINDOW008_FIX01_SOURCE_MANIFEST.json', 114)]:
        path = ROOT / name
        manifest = json.loads(path.read_text('utf8'))
        differences = []
        for relative, expected in manifest['files'].items():
            file = ROOT / relative
            actual = sha(file) if file.is_file() else None
            if actual != expected:
                differences.append(dict(path=relative, expected=expected, actual=actual))
        manifests[name] = dict(sha256=sha(path), file_count=len(manifest['files']),
                               expected_count=expected_count, differences=differences,
                               passed=len(manifest['files']) == expected_count and not differences)
    config_path = ROOT / 'window008_resume01_config.json'
    config = json.loads(config_path.read_text('utf8'))
    references = {}
    for key, reference in config['reference_inputs'].items():
        path = (ROOT / reference['path']).resolve()
        actual = sha(path) if path.is_file() else None
        passed = path.is_relative_to(ROOT) and actual == reference['sha256'] and path.stat().st_size == reference['bytes']
        references[key] = dict(path=reference['path'], bytes=path.stat().st_size if path.is_file() else None,
                                expected_sha256=reference['sha256'], actual_sha256=actual, passed=passed)
    config_links = {}
    for field in ['base_configuration', 'legacy_source_manifest', 'fix01_configuration', 'fix01_source_manifest']:
        config_links[field] = sha(ROOT / config[field]) == config[field + '_sha256']
    review = dict(schema='n01/window008-resume01-independent-source-scope/v1',
                  UTC=datetime.now(timezone.utc).isoformat(),
                  status='passed' if all(row['passed'] for row in manifests.values())
                       and all(row['passed'] for row in references.values()) and all(config_links.values()) else 'failed',
                  config_sha256=sha(config_path), old_manifests=manifests,
                  reference_count=len(references), references=references, config_links=config_links,
                  scientific_model_forwards=0, optimizer_updates=0, DEV_numeric_decodes=0, GPU_calls=0,
                  raw_dataset_hashing_only=True)
    (OUT / 'IMMUTABILITY_REFERENCE_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n', 'utf8')
    print(json.dumps(dict(status=review['status'], config_sha256=review['config_sha256'],
                          manifests=manifests, reference_count=len(references), config_links=config_links), indent=2))


if __name__ == '__main__':
    main()
