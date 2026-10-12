"""Make verified private E/D backup and explicit public derived payload."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ID='noise_improvement_004_20261012'
RUN=ROOT/'runs'/ID
REC=ROOT/'receipts'/ID
REPORT=ROOT/'reports'/ID
DEST=RUN/'delivery'
BACKUP=Path('D:/GAN_backups/research/N01')/ID
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def archive(path,items):
    if path.exists():raise FileExistsError(path)
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in items:z.write(p,name)
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Archive CRC mismatch: '+bad)
        return len(z.infolist())
def main():
    if DEST.exists():raise FileExistsError('Preserve existing delivery')
    final=json.loads((REC/'FINAL_RECEIPT.json').read_text(encoding='utf8'))
    assert final['noise_training_updates']==5000 and final['response_training_updates']==0
    assert json.loads((REC/'VISUAL_QA.json').read_text(encoding='utf8'))['status']=='passed'
    DEST.mkdir(parents=True);BACKUP.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    bundle=DEST/'n01-noise-improvement-004-source.bundle'
    subprocess.run(['git','-C',str(ROOT),'bundle','create',str(bundle),'HEAD'],check=True)
    subprocess.run(['git','-C',str(ROOT),'bundle','verify',str(bundle)],check=True,capture_output=True)
    items=[]
    for folder,label in [(RUN/'runs','runs'),(REPORT,'report'),(REC,'receipts')]:
        items.extend((p,label+'/'+p.relative_to(folder).as_posix()) for p in sorted(folder.rglob('*')) if p.is_file())
    items.extend([(RUN/'EXECUTION_SOURCE.tar.gz','EXECUTION_SOURCE.tar.gz'),(bundle,bundle.name)])
    items.extend((p,'visual_QA/'+p.name) for p in sorted(RUN.glob('HTML_QA_*.png')))
    private=DEST/'n01-noise-improvement-004-private-full.zip'
    members=archive(private,items)
    public_items=[]
    science=RUN/'runs/science'
    # Everything public is named explicitly; REAL baseline_populations is excluded.
    for name in ['final.pt','initial_state.pt','replay.pt','training.jsonl','fit_statistics.json','TRAINING_COMPLETE.json']:
        p=science/'noise'/name
        if p.exists():public_items.append((p,'noise/'+name))
    for name in ['NOISE_GATE.json','DESCRIPTIVE_METRICS.json']:
        p=science/'noise_validation'/name;public_items.append((p,'noise_validation/'+name))
    for p in sorted((science/'noise_validation/classifiers').rglob('*')):
        if p.is_file() and p.name in ['classifier.pt','predictions.npz','result.json']:
            public_items.append((p,'noise_validation/classifiers/'+p.relative_to(science/'noise_validation/classifiers').as_posix()))
    for folder,label in [(REPORT,'report'),(REC,'receipts'),(RUN/'runs/replay','replay')]:
        public_items.extend((p,label+'/'+p.relative_to(folder).as_posix()) for p in sorted(folder.rglob('*')) if p.is_file())
    public_items.extend((p,'visual_QA/'+p.name) for p in sorted(RUN.glob('HTML_QA_*.png')))
    public_items.extend((science/name,'science/'+name) for name in ['COMPLETE.json','exit_receipt.json','launch.json','status.json'] if (science/name).exists())
    supervisor=RUN/'runs/supervisor'
    public_items.extend((supervisor/name,'supervisor/'+name) for name in ['exit_receipt.json','launch.json','resource_status.json'] if (supervisor/name).exists())
    inventory=DEST/'PUBLIC_INVENTORY.json'
    write(inventory,dict(run_id=ID,commit=commit,contains_raw_REAL_waveforms=False,
        excluded=['noise_validation/baseline_populations.npz','private REAL inputs','private complete backup'],
        files=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for p,name in public_items]))
    public_items.append((inventory,'PUBLIC_INVENTORY.json'))
    public=DEST/'n01-noise-improvement-004-public-derived.zip'
    pub_members=archive(public,public_items)
    sums=DEST/'PUBLIC_SHA256SUMS.txt'
    sums.write_text(''.join(sha(p)+'  '+p.name+'\n' for p in [public,bundle]),encoding='ascii')
    backups=[]
    for p in [private,public,bundle,sums,inventory]:
        q=BACKUP/p.name;shutil.copy2(p,q);assert sha(p)==sha(q)
        backups.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p),D_copy=str(q),D_verified=True))
    receipt=dict(status='E_D_archives_reopened_and_sha256_verified',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        commit=commit,private_members=members,public_members=pub_members,private_backup=str(BACKUP/private.name),
        public_REAL_waveform_arrays=False,files=backups)
    write(REC/'BACKUP_RECEIPT.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
