"""Push the explicit publication closure and verify the final remote delivery."""
import base64
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT/'receipts/mechanism012_publication_20261015'
REPO = 'Aggressivecyber/ej276d-n01'
BASE = 'c432920776720dc2ca4cb63f94fb2db13112cd8b'
REMOTE_START = 'b59394013891c1daa1d4ae874083a8c6dfcb3d83'
POINTER = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))


def execute(args):
    result = subprocess.run(args,cwd=ROOT,capture_output=True,encoding='utf-8',timeout=60)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.stderr}')
    return result.stdout.strip()


def git(*args):
    return execute(['git','-c','core.autocrlf=false',*args])


def api(path):
    return json.loads(execute(['gh','api',path]))


def main():
    assert git('rev-parse','HEAD') == BASE
    assert not git('diff','--cached','--name-only')
    assert set(git('diff','--name-only').splitlines()) == {'README.md','PROGRAM.json','jobs.json'}
    publication = read(OUT/'PUBLICATION_RECEIPT.json')
    assert publication['verified_remote_source_commit'] == BASE
    assert api(f'repos/{REPO}/git/ref/heads/main')['object']['sha'] == BASE
    selected = [ROOT/'README.md',ROOT/'PROGRAM.json',ROOT/'jobs.json',
        ROOT/'receipts/mechanism012_20261015/LOCAL_FINAL_DELIVERY_RECEIPT.json',
        OUT/'prepare_publication_closeout.py',OUT/'complete_github_delivery.py',
        OUT/'PUBLICATION_RECEIPT.json',OUT/'LINE_ENDING_READBACK.json',OUT/'BYTE_PRESERVATION_RECEIPT.json',
        OUT/'SELECTED_COMMIT_FILES.json',OUT/'COMMIT_MESSAGE.txt',OUT/'BYTE_PRESERVATION_COMMIT_MESSAGE.txt',
        OUT/'initial_manifest_readback_failure_01/FAILURE.json',
        OUT/'initial_manifest_readback_failure_01/prepare_publication_closeout.py']
    message = (OUT/'COMMIT_MESSAGE.txt').read_text('utf-8')
    assert 'Publish N01' in message
    readme = (ROOT/'README.md').read_text('utf-8')
    assert readme.startswith('# N01：012') and '2163' in readme and 'response_match_not_established' in readme
    for label in ['reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md',
        'reports/shape010_20261014/N01_SHAPE010_REPORT.md',
        'receipts/stability011_20261014/user_server_pause_01/native_result_json/STABILITY_RESULT.json',
        'docs/MEAN_RANDOM009_RESULTS.md']:
        assert (ROOT/label).is_file() and label in readme
    for path in selected:
        assert path.is_file() and path.is_relative_to(ROOT)
        assert path.suffix in {'.json','.md','.py','.txt'}
    record = dict(schema='n01/mechanism012-publication-final-selection/v1',UTC=datetime.now(timezone.utc).isoformat(),
        parent_commit=BASE,files=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in selected],
        raw_scientific_payloads=False,other_untracked_preserved=True)
    selection_path = OUT/'SELECTED_DELIVERY_COMMIT_FILES.json'
    assert not selection_path.exists()
    write(selection_path,record)
    selected.append(selection_path)
    relative = [p.relative_to(ROOT).as_posix() for p in selected]
    git('add','--',*relative)
    changed = set(git('diff','--cached','--name-only').splitlines())
    assert changed == set(relative)
    git('commit','--only','--file',str(OUT/'COMMIT_MESSAGE.txt'),'--',*relative)
    tip = git('rev-parse','HEAD')
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    for manifest_name,count in [('MECHANISM012_SOURCE_MANIFEST.json',69),('STABILITY011_SOURCE_MANIFEST.json',63),('SHAPE010_SOURCE_MANIFEST.json',53)]:
        manifest = read(ROOT/manifest_name)
        assert len(manifest['files']) == count
        for relative_path,wanted in manifest['files'].items():
            assert sha(ROOT/relative_path) == wanted
            blob = subprocess.check_output(['git','cat-file','blob',f'{tip}:{relative_path}'],cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest() == wanted
    push = subprocess.run(['git','-c','http.version=HTTP/1.1','push','origin','HEAD:main'],cwd=ROOT,
        capture_output=True,encoding='utf-8',timeout=60)
    write(OUT/'FINAL_PUSH_RECEIPT.json',dict(schema='n01/mechanism012-git-push/v1',UTC=datetime.now(timezone.utc).isoformat(),
        local_commit=tip,exit_code=push.returncode,stdout=push.stdout,stderr=push.stderr,force_push=False))
    if push.returncode:
        raise RuntimeError('publication closure push unconfirmed; inspect remote before retry')
    remote = api(f'repos/{REPO}/git/ref/heads/main')['object']['sha']
    assert remote == tip
    assert git('ls-remote','origin','refs/heads/main').split()[0] == tip
    verified = []
    critical = ['README.md','PROGRAM.json','jobs.json','receipts/mechanism012_publication_20261015/PUBLICATION_RECEIPT.json',
        'reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md','MECHANISM012_SOURCE_MANIFEST.json','MECHANISM012_BINDINGS.json']
    for name in critical:
        payload = api(f'repos/{REPO}/contents/{name}?ref={tip}')
        assert payload['encoding'] == 'base64'
        raw = base64.b64decode(payload['content'])
        assert raw == (ROOT/name).read_bytes(), name
        verified.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'GitHub_blob_sha':payload['sha']})
    program = read(ROOT/'PROGRAM.json')
    pointer = read(POINTER)
    assert program['mechanism012_publication_status'] == pointer['mechanism012_publication_status'] == 'github_main_committed_and_report_SHA256_verified'
    assert program['active_scientific_jobs'] == pointer['active_scientific_jobs'] == 0
    assert program['new_training_updates'] == pointer['new_training_updates'] == 159667
    assert program['scientific_status'] == pointer['scientific_status'] == 'response_match_not_established'
    delivery = dict(schema='n01/mechanism012-GitHub-final-delivery/v1',UTC=datetime.now(timezone.utc).isoformat(),
        status='GitHub_main_tip_and_final_critical_file_bytes_verified',repository=REPO,branch='main',
        latest_commit=tip,latest_commit_url=f'https://github.com/{REPO}/commit/{tip}',
        source_result_commit='dc1b55a3a79308af0699a483f709d7104fdbd133',byte_preservation_commit=BASE,
        remote_start=REMOTE_START,pushed_commit_count=len(git('rev-list',f'{REMOTE_START}..{tip}').splitlines()),
        remote_git_ref=remote,remote_GitHub_API_ref=remote,verified_files=verified,
        report_url=f'https://github.com/{REPO}/blob/{tip}/reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md',
        publication_receipt_sha256=sha(OUT/'PUBLICATION_RECEIPT.json'),
        current_pointer_sha256=sha(POINTER),original_local_delivery_receipt_preserved=True,
        frozen_source_Git_blob_guards={'012':69,'011':63,'010':53},
        tracked_worktree_clean=True,other_untracked_preserved=True,active_scientific_jobs=0,
        new_training_updates=0,new_NN_rows=0,new_AD_queries=0,checkpoint_loads=0,
        scientific_status='response_match_not_established',formal_response_acceptance=False)
    delivery_path = OUT/'FINAL_DELIVERY_RECEIPT.json'
    assert not delivery_path.exists()
    write(delivery_path,delivery)
    backup = Path('D:/GAN_backups/N01/mechanism012_20261015/publication_01')
    backup.mkdir(parents=True,exist_ok=False)
    backup_files = [OUT/'PUBLICATION_RECEIPT.json',OUT/'BYTE_PRESERVATION_RECEIPT.json',OUT/'LINE_ENDING_READBACK.json',
        OUT/'FINAL_PUSH_RECEIPT.json',delivery_path,ROOT/'README.md',ROOT/'PROGRAM.json',ROOT/'jobs.json',
        POINTER,ROOT/'.gitattributes']
    sums = []
    for source in backup_files:
        target = backup/source.name
        assert not target.exists()
        target.write_bytes(source.read_bytes())
        assert sha(target) == sha(source)
        sums.append(f'{sha(target)}  {target.name}')
    (backup/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n',encoding='utf-8')
    assert read(delivery_path)['remote_GitHub_API_ref'] == tip
    print(json.dumps({'status':delivery['status'],'commit':tip,'pushed_commits':delivery['pushed_commit_count'],
        'report_url':delivery['report_url'],'final_receipt':str(delivery_path),'final_receipt_sha256':sha(delivery_path),
        'D_publication_backup':str(backup)},ensure_ascii=False))


if __name__ == '__main__':
    main()
