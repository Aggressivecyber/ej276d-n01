"""Package frozen Window008 public artifacts; no training or model forward."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def archive(path, files, extra=None):
    hashes = {name: sha(source) for name, source in sorted(files.items())}
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, source in sorted(files.items()):
            z.write(source, name)
        for name, value in (extra or {}).items():
            assert name not in hashes
            z.writestr(name, value)
            hashes[name] = hashlib.sha256(value.encode('utf8')).hexdigest()
        z.writestr('SHA256SUMS.txt', ''.join(f'{digest}  {name}\n' for name, digest in sorted(hashes.items())))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(hashes) | {'SHA256SUMS.txt'}
        for name, digest in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
    return {'bytes': path.stat().st_size, 'sha256': sha(path), 'verified_members': len(hashes) + 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    if sys.platform == 'win32' and out.drive.upper() not in ('E:', 'D:'):
        raise ValueError('Publication output must be on E: or D:')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'weights').mkdir()
    sys.path.insert(0, str(root / 'src'))
    import torch
    from n01.model008 import N01Response008
    from n01.model008_fix01 import N01Response008Fix01
    from n01.electronics_v3 import ObservedBaselineNoiseV3
    from n01.response008 import state_digest
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)

    manifest_names = ['WINDOW008_RESUME01_SOURCE_MANIFEST.json',
                      'WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json',
                      'WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json']
    source_files, withheld = {}, {}
    for name in manifest_names:
        source_files[name] = root / name
        for rel, digest in read(root / name)['files'].items():
            path = root / rel
            assert path.resolve().is_relative_to(root)
            assert sha(path) == digest, rel
            if path.suffix.lower() in ('.py', '.sh', '.md', '.json'):
                source_files[rel] = path
            else:
                withheld[rel] = {'sha256': digest, 'reason': 'binary training input or full checkpoint; not in frozen text source archive'}
    source_note = {
        'schema': 'n01/window008-public-source-scope/v1',
        'source_manifests_sha256': {name: sha(root / name) for name in manifest_names},
        'scope': 'Exact frozen text files only; this archive is not the full private execution payload.',
        'withheld_binary_inputs': withheld,
    }
    assets = {}
    name = 'N01_WINDOW008_RESUME01_FROZEN_SOURCE.zip'
    assets[name] = archive(out / name, source_files, {
        'PUBLIC_SOURCE_SCOPE.json': json.dumps(source_note, ensure_ascii=False, indent=2) + '\n'})

    report = root / 'reports/window008_resume01_20261013'
    build = read(report / 'BUILD_RECEIPT.json')
    report_files = {}
    for rel, digest in build['files'].items():
        path = report / rel
        assert path.resolve().is_relative_to(report)
        assert sha(path) == digest, rel
        report_files[rel] = path
    for rel in ('BUILD_RECEIPT.json', 'PDF_TEXT.txt', 'TEXT_REOPEN_QA.json'):
        report_files[rel] = report / rel
    for rel in ('REPORT_VISUAL_QA.json', 'FINAL_DELIVERY_RECEIPT.json'):
        report_files[rel] = root / 'receipts/window008_resume01_20261013' / rel
    name = 'N01_WINDOW008_RESUME01_REPORT_EVIDENCE.zip'
    assets[name] = archive(out / name, report_files)
    pdf_name = 'N01_WINDOW008_RESUME01_REPORT.pdf'
    shutil.copyfile(report / pdf_name, out / pdf_name)
    assert sha(out / pdf_name) == build['files'][pdf_name]
    assets[pdf_name] = {'bytes': (out / pdf_name).stat().st_size, 'sha256': sha(out / pdf_name)}

    completion = read(root / 'runs/window008_resume01_20261013/runs/science/TRAINING_COMPLETE.json')
    cfg = read(root / 'window008_config.json')
    statistics = read(root / cfg['statistics'])
    model_statistics = statistics['model']
    weight_files, weight_records = {}, {}
    for arm, endpoint in completion['endpoint_map'].items():
        path = root / endpoint['path'] if arm == 'deterministic' else root / 'runs/window008_resume01_20261013' / endpoint['path']
        assert sha(path) == endpoint['sha256'], arm
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['arm'] == arm and state['G_updates'] == 19000
        assert state['statistics']['model'] == model_statistics
        weights = {k: v.detach().cpu().clone() for k, v in state['model'].items()}
        assert state_digest(weights) == state_digest(state['model'])
        public = out / 'weights' / f'{arm}.state_dict.pt'
        torch.save(weights, public)
        reopened = torch.load(public, map_location='cpu', weights_only=True)
        assert state_digest(reopened) == state_digest(state['model'])
        model = (N01Response008 if arm == 'deterministic' else N01Response008Fix01)(model_statistics, arm)
        model.load_state_dict(reopened, strict=True)
        assert state_digest(model.state_dict()) == state_digest(state['model'])
        weight_files[public.name] = public
        weight_records[arm] = {
            'file': public.name, 'sha256': sha(public), 'model_class': endpoint['model_implementation'],
            'source_checkpoint_sha256': endpoint['sha256'], 'source_checkpoint_path': str(path.relative_to(root)).replace('\\', '/'),
            'source_schema': endpoint['schema'], 'G_updates': state['G_updates'],
            'state_sha256': state_digest(weights), 'tensor_count': len(weights),
            'weights_only_reopen': True, 'strict_model_state_load': True, 'forward_calls': 0,
        }
        del model, state, weights, reopened
    public_stats = out / 'weights/MODEL_STATISTICS.json'
    write(public_stats, model_statistics)
    weight_files[public_stats.name] = public_stats

    noise_path = root / cfg['noise_reference']['path']
    assert sha(noise_path) == cfg['noise_reference']['sha256']
    noise = torch.load(noise_path, map_location='cpu', weights_only=False)
    init_keys = ('length', 'mean_V', 'cholesky_V', 'voltage_scale', 'normalized_diagonal_floor', 'initial_log_scale_std')
    def plain(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        return value
    initialization = {k: plain(noise['initialization'][k]) for k in init_keys}
    weights = {k: v.detach().cpu().clone() for k, v in noise['model'].items()}
    public = out / 'weights/noise005.state_dict.pt'
    torch.save(weights, public)
    reopened = torch.load(public, map_location='cpu', weights_only=True)
    model = ObservedBaselineNoiseV3(initialization)
    model.load_state_dict(reopened, strict=True)
    assert state_digest(model.state_dict()) == state_digest(noise['model'])
    init_path = out / 'weights/NOISE005_INITIALIZATION.json'
    write(init_path, initialization)
    weight_files.update({public.name: public, init_path.name: init_path})
    weight_records['noise005'] = {
        'file': public.name, 'sha256': sha(public), 'model_class': 'n01.electronics_v3.ObservedBaselineNoiseV3',
        'source_checkpoint_sha256': sha(noise_path), 'state_sha256': state_digest(weights),
        'tensor_count': len(weights), 'weights_only_reopen': True, 'strict_model_state_load': True,
        'reused_frozen_noise_model': True, 'forward_calls': 0,
    }
    weights_manifest = out / 'weights/WEIGHTS_MANIFEST.json'
    write(weights_manifest, {'schema': 'n01/window008-public-inference-weights/v1', 'models': weight_records,
        'model_statistics_sha256': sha(public_stats), 'noise_initialization_sha256': sha(init_path),
        'optimizer_and_sampler_state_included': False, 'training_resume_checkpoint': False,
        'scientific_status': 'response_match_not_established'})
    weight_files[weights_manifest.name] = weights_manifest
    name = 'N01_WINDOW008_RESUME01_INFERENCE_WEIGHTS.zip'
    assets[name] = archive(out / name, weight_files, {'README.md':
        '# Window008 inference state dictionaries\n\n'
        'Load tensors with torch.load(path, map_location="cpu", weights_only=True).\n'
        'Use the exact frozen source archive and MODEL_STATISTICS.json.\n'
        'Construct N01Response008(stats, "deterministic") or N01Response008Fix01(stats, arm), '
        'then load_state_dict(state, strict=True), requires_grad_(False), and eval().\n'
        'The reused noise005 module uses ObservedBaselineNoiseV3(NOISE005_INITIALIZATION.json).\n'
        'Its 100-point baseline is extended by the frozen response008/observation008 pipeline; '
        'fast/slow/common_level remain distinct assumptions. Preserve the recorded baseline, ADC, peak, and 256-point crop operations.\n\n'
        'All exports were reopened with weights_only=True and strictly loaded; tensor digests match the original endpoints. '
        'Packaging performed zero model forwards and zero optimizer updates. '
        'These are inference weights, not resumable training checkpoints. Physical response matching is not established.\n'})

    manifest = {
        'schema': 'n01/window008-publication-package/v1', 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'release_tag': 'n01-window008-resume01-20261014',
        'source_commit_before_publication': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'user_authorization': '2026-10-14: 上传github',
        'scientific_status': 'response_match_not_established',
        'source_manifests_sha256': source_note['source_manifests_sha256'],
        'frozen_text_file_count': len(source_files), 'withheld_binary_input_count': len(withheld),
        'assets': assets, 'weight_exports': weight_records,
        'raw_REAL_or_original_SOURCE_waveforms_included': False,
        'per_event_REAL_observables_included': False,
        'full_private_archive_included': False, 'new_optimizer_updates': 0, 'new_model_forwards': 0,
        'verification': 'All archive members reopened and SHA256-verified; all exported model tensors match original endpoints.',
    }
    manifest_path = out / 'PUBLICATION_MANIFEST.json'
    write(manifest_path, manifest)
    checksums = {name: item['sha256'] for name, item in assets.items()}
    checksums[manifest_path.name] = sha(manifest_path)
    (out / 'PUBLIC_SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(checksums.items())), encoding='utf8')
    print(json.dumps({'output': str(out), 'assets': assets, 'frozen_text_files': len(source_files),
                      'withheld_binary_inputs': len(withheld), 'verification': 'passed'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
