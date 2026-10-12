"""Verify uploaded integration006 asset digests and record public delivery."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1];ID='integration_006_20261012'
REC=ROOT/'receipts'/ID;DEST=ROOT/'runs'/ID/'delivery';TAG='n01-integration006-20261012'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    target=REC/'DELIVERY_RECEIPT.json';assert not target.exists()
    backup=read(REC/'BACKUP_RECEIPT.json');commit=backup['commit']
    release=read(DEST/'GITHUB_RELEASE.json');tag=read(DEST/'GITHUB_TAG.json')
    assert tag['object']['type']=='commit' and tag['object']['sha']==commit
    assert release['tag_name']==TAG and release['draft'] is False
    expected=['n01-integration006-public-derived.zip','n01-integration006-source.bundle','PUBLIC_SHA256SUMS.txt']
    assert sorted(a['name'] for a in release['assets'])==sorted(expected)
    assets=[]
    for name in expected:
        item=next(a for a in release['assets'] if a['name']==name);p=DEST/name
        assert item['digest']=='sha256:'+sha(p) and item['size']==p.stat().st_size and item['state']=='uploaded'
        assets.append(dict(name=name,bytes=item['size'],local_sha256=sha(p),server_digest=item['digest'],
            sha256_match=True,url=item['browser_download_url'],state=item['state']))
    main_sha=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
    assert main_sha==commit
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    r=dict(status='public_release_uploaded_and_all_asset_sha256_verified',verified_UTC=now,
        repository='https://github.com/Aggressivecyber/ej276d-n01',repository_visibility='public',
        release_url=release['html_url'],release_id=release['id'],release_tag=TAG,release_is_draft=False,
        source_and_scientific_results_commit=commit,main_at_scientific_publication_verified=main_sha,
        all_uploaded_server_digests_match=True,assets=assets,raw_or_downsampled_REAL_traces_published=False,
        per_event_REAL_observables_published=False,original_SOURCE_input_waveforms_published=False,
        private_backup_path=backup['private_backup'],archive_members_verified=backup['private_members']+backup['public_members'],
        initial_startup_failure_preserved=True,scientific_completion_receipt='FINAL_RECEIPT.json',
        delivery_metadata_may_be_committed_after_scientific_tag=True)
    write(target,r)
    p=read(ROOT/'PROGRAM.json');p.update(updated_UTC=now,integration006_scientific_results_commit=commit,
        integration006_public_release=r['release_url'],integration006_public_assets_sha256_verified=True,
        integration006_delivery_receipt='receipts/'+ID+'/DELIVERY_RECEIPT.json')
    write(ROOT/'PROGRAM.json',p)
    readme_path=ROOT/'README.md';content=readme_path.read_text(encoding='utf8')
    marker=f'[完整480点与PSD整合报告](reports/{ID}/N01_INTEGRATION006_REPORT.html)'
    assert marker in content
    content=content.replace(marker,marker+' · [006 Release]('+r['release_url']+')',1)
    readme_path.write_text(content,encoding='utf8',newline='\n')
    pointer_path=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';pointer=read(pointer_path)
    pointer.update(updated_UTC=now,plan_sha256=sha(readme_path),integration006_public_release=r['release_url'],
        integration006_delivery_receipt=str(target),integration006_public_assets_sha256_verified=True)
    write(pointer_path,pointer)
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
