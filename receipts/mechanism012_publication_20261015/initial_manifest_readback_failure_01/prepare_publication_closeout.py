"""Verify the pushed code and prepare only publication metadata and the README."""
import base64
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('E:/GAN/research/N01')
OUT = ROOT/'receipts/mechanism012_publication_20261015'
POINTER = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
REPO = 'Aggressivecyber/ej276d-n01'
SOURCE_COMMIT = 'dc1b55a3a79308af0699a483f709d7104fdbd133'
REMOTE_BEFORE = 'b59394013891c1daa1d4ae874083a8c6dfcb3d83'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args,cwd=ROOT,capture_output=True,encoding='utf-8',timeout=60)
    if result.returncode:
        raise RuntimeError(f'{args[0]} command failed: {result.stderr}')
    return result.stdout.strip()


def api(path):
    return json.loads(command(['gh','api',path]))


def write(path,value):
    data = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    temporary = path.with_suffix(path.suffix+'.publication012.tmp')
    assert not temporary.exists()
    temporary.write_bytes(data)
    temporary.replace(path)


def main():
    assert command(['git','rev-parse','HEAD']) == SOURCE_COMMIT
    assert not command(['git','diff','--name-only'])
    assert not command(['git','diff','--cached','--name-only'])
    assert api(f'repos/{REPO}/git/ref/heads/main')['object']['sha'] == SOURCE_COMMIT
    assert api('user')['login'] == 'Aggressivecyber'
    files = [
        'reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md',
        'receipts/mechanism012_20261015/postrun_independent_01/NUMERIC_READBACK.json',
        'receipts/mechanism012_20261015/postrun_independent_01/REPORT_QA_FINAL.json',
        'receipts/mechanism012_20261015/FINAL_RESULT_RECEIPT.json',
        'MECHANISM012_SOURCE_MANIFEST.json',
        'MECHANISM012_BINDINGS.json',
    ]
    verified = []
    for relative in files:
        response = api(f'repos/{REPO}/contents/{relative}?ref={SOURCE_COMMIT}')
        assert response['encoding'] == 'base64'
        remote_bytes = base64.b64decode(response['content'])
        local_bytes = (ROOT/relative).read_bytes()
        assert remote_bytes == local_bytes, relative
        verified.append(dict(path=relative,bytes=len(local_bytes),sha256=hashlib.sha256(remote_bytes).hexdigest(),
            GitHub_blob_sha=response['sha'],url=f'https://github.com/{REPO}/blob/{SOURCE_COMMIT}/{relative}'))
    before = OUT/'before_publication_01'
    before.mkdir(exist_ok=False)
    for path in [ROOT/'README.md',ROOT/'PROGRAM.json',ROOT/'jobs.json',POINTER]:
        (before/path.name).write_bytes(path.read_bytes())
    for name,count in [('MECHANISM012_SOURCE_MANIFEST.json',69),('STABILITY011_SOURCE_MANIFEST.json',63),('SHAPE010_SOURCE_MANIFEST.json',53)]:
        manifest = read(ROOT/name)
        assert len(manifest['files']) == count and 'README.md' not in manifest['files']
        for relative,expected in manifest['files'].items():
            expected_hash = expected if isinstance(expected,str) else expected['sha256']
            assert sha(ROOT/relative) == expected_hash
    utc = datetime.now(timezone.utc).isoformat()
    receipt = dict(schema='n01/mechanism012-github-publication/v1',UTC=utc,
        status='code_commit_main_and_six_critical_file_bytes_verified',repository=REPO,
        repository_url=f'https://github.com/{REPO}',branch='main',
        authenticated_login='Aggressivecyber',repository_visibility='PUBLIC',
        user_authorization='2026-10-15 Asia/Singapore: github提交',
        remote_main_before=REMOTE_BEFORE,verified_remote_source_commit=SOURCE_COMMIT,
        source_commit_url=f'https://github.com/{REPO}/commit/{SOURCE_COMMIT}',native_git_push_exit_code=0,
        pushed_existing_commit_count=5,verified_files=verified,
        transferred_scope='tracked code, protocols, reports and small verification records from009 through012',
        latest_local_delivery_receipt_sha256=sha(ROOT/'receipts/mechanism012_20261015/LOCAL_FINAL_DELIVERY_RECEIPT.json'),
        scientific_status='response_match_not_established',formal_response_acceptance=False,
        active_scientific_jobs=0,new_training_updates=0,new_NN_rows=0,new_AD_queries=0,
        checkpoint_loads=0,optimizer_steps=0,original_results_modified=False,
        immutable_source_guards_passed={'012':69,'011':63,'010':53},
        README_and_metadata_closure_commit_pending=True)
    write(OUT/'PUBLICATION_RECEIPT.json',receipt)
    headline = '''# N01：012 新头梯度与裁窗机制诊断已完成

截至 2026-10-15，009 配对诊断、010 新头固定 2000 G 试验、011 的 500 次已存 TRAIN 库配对重加权、012 局部机制诊断均已完成并读回。当前科学作业为 0；012 新增训练更新为 0，累计科学训练更新仍为 159667。

012 完成 9 次 scalar AD 和 2 次 masked endpoint VJP，A、005 和 final 新头各前向 2048 行；2163 项独立保存输出核验和 32 项报告检查通过。固定 TRAIN 点的加权 QPSD 梯度约为波形项的 19 倍；单一 carrier 方向的 0.1 LSB moving-crop 中心差分与代理梯度反号，1/10 LSB 不反号。下一项先用保存输出诊断裁窗边界，训练预算为 0。

科学状态保持 **`response_match_not_established`**；这些结果描述固定 TRAIN 库和指定方向的局部机制。

[012 诊断报告](reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md) · [012 最终结果回执](receipts/mechanism012_20261015/FINAL_RESULT_RECEIPT.json) · [独立数值读回](receipts/mechanism012_20261015/postrun_independent_01/NUMERIC_READBACK.json) · [报告检查](receipts/mechanism012_20261015/postrun_independent_01/REPORT_QA_FINAL.json) · [GitHub 核验回执](receipts/mechanism012_publication_20261015/PUBLICATION_RECEIPT.json)

[010 试验报告](reports/shape010_20261014/N01_SHAPE010_REPORT.md) · [011 配对重加权结果](receipts/stability011_20261014/user_server_pause_01/native_result_json/STABILITY_RESULT.json) · [009 结果](docs/MEAN_RANDOM009_RESULTS.md)

## 历史登记：009 平均响应与随机响应诊断
'''
    old = (before/'README.md').read_text(encoding='utf-8')
    assert old.startswith('# N01：009 平均响应与随机响应诊断\n')
    history = old.split('\n',1)[1]
    history = history.replace('当前阶段为实现、人工数值检查和源码冻结；真实首批确认后设置预计完成时单次苏醒。诊断后根据结果决定是否登记每臂 2000 更新的冻结均值修复试验。',
        '009 开始时登记的阶段为实现、人工数值检查和源码冻结；启动后安排单次苏醒。其后 010 固定 2000 G 试验与 011/012 诊断已完成，最新状态见本文顶部。',1)
    (ROOT/'README.md').write_text(headline+history,encoding='utf-8',newline='\n')
    shared = dict(updated_UTC=utc,mechanism012_publication_status='github_main_committed_and_report_SHA256_verified',
        mechanism012_public_commit=SOURCE_COMMIT,
        mechanism012_github_report=f'https://github.com/{REPO}/blob/{SOURCE_COMMIT}/{files[0]}',
        mechanism012_publication_receipt='receipts/mechanism012_publication_20261015/PUBLICATION_RECEIPT.json',
        mechanism012_publication_UTC=utc,mechanism012_github_critical_files_byte_verified=True)
    for path in [ROOT/'PROGRAM.json',POINTER]:
        value = read(path)
        assert value['status'] == 'mechanism012_closed_independently_read_back'
        assert value['scientific_status'] == 'response_match_not_established'
        assert value['active_scientific_jobs'] == 0 and value['new_training_updates'] == 159667
        value.update(shared)
        write(path,value)
    jobs = read(ROOT/'jobs.json')
    matching = [j for j in jobs['jobs'] if j.get('run_id') == 'mechanism012_20261015']
    assert len(matching) == 1
    matching[0].update(publication_status=shared['mechanism012_publication_status'],
        published_source_commit=SOURCE_COMMIT,publication_receipt=shared['mechanism012_publication_receipt'],
        github_report=shared['mechanism012_github_report'],publication_UTC=utc)
    jobs['updated_UTC'] = utc
    write(ROOT/'jobs.json',jobs)
    selected = [ROOT/'README.md',ROOT/'PROGRAM.json',ROOT/'jobs.json',
        ROOT/'receipts/mechanism012_20261015/LOCAL_FINAL_DELIVERY_RECEIPT.json',
        OUT/'prepare_publication_closeout.py',OUT/'PUBLICATION_RECEIPT.json']
    selection = dict(schema='n01/mechanism012-publication-commit-selection/v1',UTC=utc,parent_commit=SOURCE_COMMIT,
        files=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in selected],
        source_guard_unchanged=True,other_untracked_untouched=True,new_scientific_computation=0)
    write(OUT/'SELECTED_COMMIT_FILES.json',selection)
    (OUT/'COMMIT_MESSAGE.txt').write_text('Publish N01 009 through012 results and refresh the project entrypoint\n\nRecord the verified GitHub source commit and report bytes, update publication metadata, and link the completed012 diagnosis from the README. Scientific status and frozen inputs remain unchanged.\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'source_commit':SOURCE_COMMIT,'verified_files':len(verified),
        'publication_receipt':str(OUT/'PUBLICATION_RECEIPT.json'),'sha256':sha(OUT/'PUBLICATION_RECEIPT.json'),
        'metadata_and_README_prepared':True},ensure_ascii=False))


if __name__ == '__main__':
    main()
