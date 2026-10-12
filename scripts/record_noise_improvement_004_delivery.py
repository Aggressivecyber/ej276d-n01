"""Bind the actual public release to local assets and record delivery."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ID = 'noise_improvement_004_20261012'
DEST = ROOT / 'runs' / ID / 'delivery'
REC = ROOT / 'receipts' / ID
COMMIT = '98a2136b4d7dfec7359e5b4f3f61dc7752eef752'
TAG = 'n01-noise-improvement-004-20261012'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')


def main():
    receipt_path = REC / 'DELIVERY_RECEIPT.json'
    assert not receipt_path.exists(), 'Preserve existing delivery'
    release = read(DEST / 'GITHUB_RELEASE.json')
    tag = read(DEST / 'GITHUB_TAG.json')
    assert tag['object']['type'] == 'commit' and tag['object']['sha'] == COMMIT
    assert release['tag_name'] == TAG and release['draft'] is False
    expected = ['n01-noise-improvement-004-public-derived.zip', 'n01-noise-improvement-004-source.bundle', 'PUBLIC_SHA256SUMS.txt']
    assert sorted(x['name'] for x in release['assets']) == sorted(expected)
    assets = []
    for name in expected:
        asset = next(x for x in release['assets'] if x['name'] == name)
        local = DEST / name
        digest = sha(local)
        assert asset['digest'] == 'sha256:' + digest and asset['size'] == local.stat().st_size
        assert asset['state'] == 'uploaded'
        assets.append(dict(name=name, bytes=asset['size'], local_sha256=digest,
                           server_digest=asset['digest'], state=asset['state'],
                           url=asset['browser_download_url'], sha256_match=True))
    main_at_publication = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT, text=True).split()[0]
    assert main_at_publication == COMMIT
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    receipt = dict(status='public_release_uploaded_and_all_asset_sha256_verified', verified_UTC=now,
        repository='https://github.com/Aggressivecyber/ej276d-n01', repository_visibility='public',
        release_url=release['html_url'], release_id=release['id'], release_tag=TAG,
        release_is_draft=False, source_and_scientific_results_commit=COMMIT,
        main_at_scientific_publication_verified=main_at_publication,
        all_uploaded_server_digests_match=True, assets=assets,
        raw_or_downsampled_REAL_traces_published=False,
        private_backup_path=read(REC / 'BACKUP_RECEIPT.json')['private_backup'],
        archive_members_verified=142, backup_receipt='BACKUP_RECEIPT.json',
        member_verification_receipt='ARCHIVE_MEMBER_VERIFICATION.json',
        scientific_completion_receipt='FINAL_RECEIPT.json',
        delivery_metadata_may_be_committed_after_scientific_tag=True)
    write(receipt_path, receipt)
    program = read(ROOT / 'PROGRAM.json')
    program.update(updated_UTC=now, improvement_scientific_results_commit=COMMIT,
        improvement_public_release=release['html_url'], improvement_public_assets_sha256_verified=True,
        improvement_delivery_receipt='receipts/' + ID + '/DELIVERY_RECEIPT.json')
    write(ROOT / 'PROGRAM.json', program)
    readme_path = ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf8')
    marker = '本候选仅支持 100 点观测基线，不支持 480 点响应整合。'
    assert marker in readme
    readme = readme.replace(marker, marker + ' 权重、日志和复核资料已上传 [004 Release](' + release['html_url'] + ')。', 1)
    readme_path.write_text(readme, encoding='utf8', newline='\n')
    pointer_path = ROOT.parents[1] / 'CURRENT_SIM2REAL_PROGRAM.json'
    pointer = read(pointer_path)
    pointer.update(updated_UTC=now, plan_sha256=sha(readme_path),
        improvement_public_release=release['html_url'],
        improvement_delivery_receipt=str(receipt_path), improvement_public_assets_sha256_verified=True)
    write(pointer_path, pointer)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
