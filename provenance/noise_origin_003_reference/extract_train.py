from __future__ import annotations
import os,sys,time,signal,json,pathlib,datetime,zipfile,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text());limit=min(600,datetime.datetime.fromisoformat(CFG['deadline_UTC']).timestamp()-time.time()-90)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('R14 raw extraction wall limit')));signal.setitimer(signal.ITIMER_REAL,max(.001,limit))
print(json.dumps({'entry':'extract_train.py','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'maximum_s':limit}),flush=True)
import numpy as np
import interface_core as ic
ic.np=np
indices=np.frombuffer((ROOT/'new_original_index.bin').read_bytes(),dtype='<u4').astype('i8').tolist();groups=np.frombuffer((ROOT/'new_groups.bin').read_bytes(),dtype='<u2').copy()
n=len(indices);assert n==18500 and len(set(indices))==n and indices==sorted(indices) and np.array_equal(groups,np.array(indices)//50)
positions={i:j for j,i in enumerate(indices)}
a={'block_mean_V_f64':np.full((n,500),np.nan,dtype='<f8'),'window_normalized_f32':np.full((n,480),np.nan,dtype='<f4'),'window_V_f32':np.full((n,480),np.nan,dtype='<f4'),'normalization_f32':np.full((n,2),np.nan,dtype='<f4'),'geometry_i32':np.full((n,9),-1,dtype='<i4'),'signed_metrics_f64':np.full((n,8),np.nan,dtype='<f8'),'QC_reason_u16':np.zeros(n,dtype='<u2')}
cost={'reserved':dict.fromkeys(['hash','metadata','compressed','inflate'],0),'actual_returned':dict.fromkeys(['hash','metadata','compressed','inflate'],0),'seeks':0,'members':[],'hash_passes':0}
status='failed';error=None;before=None;decoded=0
def on_time(i,b):
    t=np.frombuffer(b,dtype='<f8')
    if i!=0 or len(t)!=10000 or not np.isfinite(t).all() or not np.all(np.abs(np.diff(t)-.05)<=1e-9):raise RuntimeError('inherited raw time grid')
def on_wave(i,b):
    global decoded
    decoded+=1;ic.convert_row(positions[i],b,a,None,{})
try:
    plan=CFG['remote_source'];spec=plan['container'];p=pathlib.Path(spec['path'])
    if p.is_symlink() or p.stat().st_size!=spec['bytes']:raise RuntimeError('source path/size binding')
    with p.open('rb') as raw:
        before=os.fstat(raw.fileno());f=ic.Counted(raw,cost,plan,START+limit)
        with zipfile.ZipFile(f,'r',allowZip64=True) as z:
            infos=z.infolist();wants={s['name']:s for s in plan['members']}
            if [i.filename for i in infos]!=list(wants):raise RuntimeError('inherited central order')
            for i in infos:
                s=wants[i.filename]
                if (i.file_size,i.compress_size,i.CRC,i.compress_type)!=(s['uncompressed_bytes'],s['compressed_bytes'],int(s['crc32'],16),8) or i.flag_bits not in (0,2048) or i.extract_version>45:raise RuntimeError('inherited member binding')
            ic.stream_member(f,z.getinfo('time_ns.npy'),wants['time_ns.npy'],[0],on_time)
            ic.stream_member(f,z.getinfo('waveforms.npy'),wants['waveforms.npy'],indices,on_wave)
        after=os.fstat(raw.fileno());fields=['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']
        if any(getattr(before,k)!=getattr(after,k) for k in fields):raise RuntimeError('same-handle source drift')
    assert decoded==18500
    np.savez_compressed(ROOT/'TRAIN_ADDITION.npz',waveforms_V=a['window_V_f32'],QC_reason_u16=a['QC_reason_u16'],original_index=np.array(indices,dtype='<u4'),artificial_group=groups,geometry_i32=a['geometry_i32'],normalization_f32=a['normalization_f32'],signed_metrics_f64=a['signed_metrics_f64'])
    status='complete'
except BaseException as ex:
    error={'error':repr(ex),'traceback':traceback.format_exc()};raise
finally:
    qc=a['QC_reason_u16'];receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':status,'error':error,'cost':cost,'new_raw_rows_decoded':decoded,'new_raw_values_decoded':decoded*10000,'time_values_decoded':10000,'new_QC_valid':int((qc==0).sum()) if status=='complete' else None,'QC_failure_bit_counts':{k:int(((qc&v)!=0).sum()) for k,v in ic.QC.items()},'QC_reasons_definition':ic.QC,'whole_source_hash_passes':0,'inherited_source_SHA256':CFG['remote_source']['container']['sha256'],'unselected_numeric_decodes':0,'EVAL_numeric_decode':0,'CAL_numeric_decode':0,'DEV_numeric_decode':0,'source_identity_string_decodes':0,'raw_long_waveforms_saved':False,'migration_content_tests':False,'peak_RSS_bytes':ic.peak_rss(),'NumPy':np.__version__,'python':sys.version}
    (ROOT/'EXTRACTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
