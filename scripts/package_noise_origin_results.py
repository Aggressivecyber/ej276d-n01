"""Package this completed audit on D, separating private traces from public derivatives."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='noise_origin_audit_003_20261012'
DEST=Path('D:/GAN_backups/research/N01')/NAME

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def archive(path,files,commit,public):
    if path.exists():raise FileExistsError(path)
    manifest={'experiment_id':NAME,'source_commit':commit,'public':public,
              'contains_raw10000_traces':False,'contains_downsampled500_traces':not public,
              'raw_source_archive_not_copied':True,'files':{}}
    for name,p in sorted(files.items()):manifest['files'][name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise RuntimeError('Archive CRC failure')
        stored=json.loads(z.read('PACKAGE_MANIFEST.json'))
        if stored!=manifest:raise RuntimeError('Manifest roundtrip mismatch')
        if len(z.namelist())!=len(files)+1:raise RuntimeError('Archive file count mismatch')
        for name,info in manifest['files'].items():
            with z.open(name) as f:
                h=hashlib.sha256();size=0
                for b in iter(lambda:f.read(1024**2),b''):h.update(b);size+=len(b)
            if h.hexdigest()!=info['sha256'] or size!=info['bytes']:raise RuntimeError('Archive member mismatch: '+name)
        if public and any(Path(name).name=='BLOCK_MEANS.npz' for name in z.namelist()):raise RuntimeError('Private trace in public package')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'members':len(files)+1,
            'all_members_reopened_sha256_verified':True,'contains_downsampled500_traces':not public}

def main():
    if git('status','--porcelain').strip():raise RuntimeError('Commit all reviewable source before packaging')
    if shutil.disk_usage(DEST.parent).free<600*1024**2:raise RuntimeError('Insufficient bounded D backup free space')
    DEST.mkdir(exist_ok=True)
    commit=git('rev-parse','HEAD').decode().strip()
    names=git('ls-files','-z').decode('utf8').strip('\x00').split('\x00')
    files={name:ROOT/name for name in names}
    for relative in [f'runs/{NAME}',f'receipts/{NAME}']:
        for p in (ROOT/relative).rglob('*'):
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=p
    files['data/PARENT_IDENTITIES.npz']=ROOT/'data/PARENT_IDENTITIES.npz'
    private=archive(DEST/'n01-noise-origin-003-private-full.zip',files,commit,False)
    public_files={name:p for name,p in files.items() if p.name!='BLOCK_MEANS.npz'}
    assert len(files)-len(public_files)==1
    public=archive(DEST/'n01-noise-origin-003-public-derived.zip',public_files,commit,True)
    bundle=DEST/'n01-noise-origin-003-source.bundle'
    if bundle.exists():raise FileExistsError(bundle)
    subprocess.run(['git','-C',str(ROOT),'bundle','create',str(bundle),'HEAD'],check=True,capture_output=True)
    verified=subprocess.run(['git','-C',str(ROOT),'bundle','verify',str(bundle)],check=True,capture_output=True,text=True)
    source={'path':str(bundle),'bytes':bundle.stat().st_size,'sha256':sha(bundle),'git_bundle_verified':True,'commit':commit}
    summary={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'backups_reopened_and_verified','experiment_id':NAME,'source_commit':commit,'private_full':private,'public_derived':public,'source_bundle':source,'public_excludes':['runs/'+NAME+'/runs/science/origin/BLOCK_MEANS.npz'],'preserved_prior_backups':True,'D_free_bytes_after':shutil.disk_usage(DEST).free}
    write(DEST/'BACKUP_RECEIPT.json',summary)
    write(ROOT/'receipts'/NAME/'BACKUP_RECEIPT.json',summary)
    lines=[v['sha256']+'  '+Path(v['path']).name for v in [private,public,source]]
    (DEST/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (DEST/'PUBLIC_SHA256SUMS.txt').write_text('\n'.join(lines[1:])+'\n',encoding='utf8')
    print(json.dumps(summary))

if __name__=='__main__':main()
