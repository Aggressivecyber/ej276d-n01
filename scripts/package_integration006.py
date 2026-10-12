"""Preserve the full local006 run and publish only an explicit derived allowlist."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1];ID='integration_006_20261012'
RUN=ROOT/'runs'/ID;REC=ROOT/'receipts'/ID;REPORT=ROOT/'reports'/ID
DEST=RUN/'delivery';BACKUP=Path('D:/GAN_backups/research/N01')/ID
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for value in iter(lambda:f.read(1024**2),b''):h.update(value)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def archive(path,items):
    assert not path.exists(),path
    names=[name for _,name in items];assert len(names)==len(set(names))
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in items:z.write(p,name)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for p,name in items:assert hashlib.sha256(z.read(name)).hexdigest()==sha(p),name
    return len(items)
def folder_items(folder,label):return [(p,label+'/'+p.relative_to(folder).as_posix()) for p in sorted(folder.rglob('*')) if p.is_file()]

def main():
    assert not DEST.exists() and not BACKUP.exists(),'Preserve existing deliveries and backups'
    final=read(REC/'FINAL_RECEIPT.json');assert final['updates']==dict(noise=0,response=0,classifier=0,GMM_fits=0)
    assert read(REC/'VISUAL_QA.json')['status']=='passed'
    assert shutil.disk_usage(BACKUP.parent).free>400*1024**2,'D backup requires400MiB free; no automatic deletion'
    DEST.mkdir(parents=True);BACKUP.mkdir(parents=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    bundle=DEST/'n01-integration006-source.bundle'
    subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,capture_output=True)
    private_items=folder_items(RUN/'runs','runs')+folder_items(RUN/'binding_fix01','binding_fix01')+folder_items(RUN/'id_fix02','id_fix02')+folder_items(REPORT,'report')+folder_items(REC,'receipts')
    private_items.extend([(RUN/'EXECUTION_SOURCE.tar.gz','EXECUTION_SOURCE.tar.gz'),(bundle,bundle.name)])
    private_items.extend(folder_items(RUN/'visual_QA','visual_QA'))
    private=DEST/'n01-integration006-private-full.zip';private_members=archive(private,private_items)
    science=RUN/'id_fix02/runs/science'
    public_items=folder_items(REPORT,'reports/'+ID)+folder_items(REC,'receipts/'+ID)+folder_items(RUN/'id_fix02/runs/verification_adc_fix02','verification')
    public_items.extend(folder_items(RUN/'visual_QA','visual_QA'))
    # No original REAL/SOURCE input tensors or per-event REAL observable arrays.
    for name in ('curves.npz','theory.npz','bootstrap.npz','sample_waveforms.pt','SUMMARY.json','DECISION.json','COMPLETE.json'):
        public_items.append((science/name,'science/'+name))
    for name in ('integration006_config.json','INTEGRATION006_SOURCE_MANIFEST.json','INTEGRATION006_BINDING_FIX01_SOURCE_MANIFEST.json','INTEGRATION006_ID_FIX02_SOURCE_MANIFEST.json','docs/INTEGRATION_006_PROTOCOL.md','docs/INTEGRATION_006_BINDING_FIX01.md','docs/INTEGRATION_006_ID_FIX02.md','docs/INTEGRATION_006_ID_FIX02_VERIFIER_ADDENDUM.md','docs/INTEGRATION_006_REPLAY_NUMERIC_FIX01.md','docs/INTEGRATION_006_REPLAY_ADC_FIX02.md'):
        public_items.append((ROOT/name,name))
    refs=ROOT/'provenance/integration006_reference'
    public_items.extend(folder_items(refs,'provenance/integration006_reference'))
    supervisor=RUN/'id_fix02/runs/supervisor'
    for name in ('launch.json','exit_receipt.json','resource_status.json'):
        if (supervisor/name).is_file():public_items.append((supervisor/name,'supervisor/'+name))
    inventory=DEST/'PUBLIC_INVENTORY.json'
    write(inventory,dict(run_id=ID,commit=commit,contains_raw_REAL_waveforms=False,contains_original_SOURCE_waveforms=False,
        REAL_aggregate_curves_only=True,full_input_reproduction_requires_private_bound_container=True,
        excluded=['observables.npz (includes per-event REAL readouts)','replay128.pt (original SOURCE inputs)',
            'latents/*.pt (original SOURCE replay inputs)','data/PARENT_IDENTITIES.npz','private full backup','private TRAIN container'],
        files=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for p,name in public_items]))
    public_items.append((inventory,'PUBLIC_INVENTORY.json'))
    public=DEST/'n01-integration006-public-derived.zip';public_members=archive(public,public_items)
    sums=DEST/'PUBLIC_SHA256SUMS.txt';sums.write_text(''.join(sha(p)+'  '+p.name+'\n' for p in (public,bundle)),encoding='ascii')
    files=[]
    for p in (private,public,bundle,sums,inventory):
        target=BACKUP/p.name;shutil.copy2(p,target);assert sha(p)==sha(target)
        files.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p),D_copy=str(target),D_verified=True))
    receipt=dict(status='E_D_archives_all_members_reopened_and_sha256_verified',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        commit=commit,private_members=private_members,public_members=public_members,private_backup=str(BACKUP/private.name),
        public_REAL_waveform_arrays=False,public_per_event_REAL_observables=False,files=files)
    write(REC/'BACKUP_RECEIPT.json',receipt);print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
