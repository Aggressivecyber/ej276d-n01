from __future__ import annotations
import ast,datetime,hashlib,json,math,os,pathlib,struct,time,zipfile,zlib
try:
 import resource
except ImportError:
 resource=None
np=None
V_MIN,V_MAX,LEVELS=-.5,.5,16383
def peak_rss():
    if resource is not None:return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    import ctypes
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    counters=PMC();counters.cb=ctypes.sizeof(counters)
    process=ctypes.windll.kernel32.GetCurrentProcess
    process.restype=wintypes.HANDLE
    get=ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    if not get(process(),ctypes.byref(counters),counters.cb):raise OSError('peak working set unavailable')
    return counters.PeakWorkingSetSize

class Counted:
    def __init__(self,f,io_cost,plan,until):self.f=f;self.cost=io_cost;self.plan=plan;self.until=until
    def check(self):
        if time.monotonic()>=self.until:raise TimeoutError('worker deadline')
        if peak_rss()>self.plan['memory_limit_bytes']:raise MemoryError('512MiB RSS target exceeded')
    def reserve(self,key,n):
        self.check()
        if type(n) is not int or n<0:raise RuntimeError('reservation integer')
        caps=self.plan['io_caps'];c=self.cost
        if c['reserved'][key]+n>caps[key]:raise RuntimeError('BEFORE_IO_'+key)
        if key in ('hash','metadata','compressed') and sum(c['reserved'][k] for k in ('hash','metadata','compressed'))+n>caps['source_total']:raise RuntimeError('BEFORE_IO_source_total')
        c['reserved'][key]+=n
    def read(self,n=-1):
        if n is None or n<0:n=self.plan['container']['bytes']-self.f.tell()
        self.reserve('metadata',n)
        return self.returned_read('metadata',n)
    def returned_read(self,key,n):
        self.check()
        try:b=self.f.read(n);self.cost['actual_returned'][key]+=len(b);return b
        except BaseException:
            self.cost['io_exception_extra_consumption']='unknown beyond returned bytes; reservation retained';raise
    def seek(self,x,w=0):self.check();self.cost['seeks']+=1;return self.f.seek(x,w)
    def tell(self):return self.f.tell()
    def seekable(self):return True

def local_header(f,info):
    f.seek(info.header_offset);b=f.read(30)
    if len(b)!=30:raise RuntimeError('short local header')
    sig,v,flags,method,tm,dt,crc,cs,us,nl,el=struct.unpack('<IHHHHHIIIHH',b)
    if sig!=0x04034b50 or v>45 or flags!=info.flag_bits or method!=8 or crc!=info.CRC or nl>256 or el>256:raise RuntimeError('local header binding')
    name=f.read(nl);extra=f.read(el)
    if len(name)!=nl or len(extra)!=el or name.decode('utf-8' if flags&2048 else 'cp437')!=info.filename:raise RuntimeError('local filename/extra')
    pos=0;z64=None
    while pos<len(extra):
        if pos+4>len(extra):raise RuntimeError('extra truncated')
        tag,n=struct.unpack('<HH',extra[pos:pos+4]);pos+=4
        if pos+n>len(extra):raise RuntimeError('extra payload truncated')
        if tag==1:
            if z64 is not None:raise RuntimeError('duplicate ZIP64')
            z64=extra[pos:pos+n]
        pos+=n
    off=0
    if us==0xffffffff:
        if z64 is None or off+8>len(z64):raise RuntimeError('ZIP64 uncompressed size')
        us=struct.unpack('<Q',z64[off:off+8])[0];off+=8
    if cs==0xffffffff:
        if z64 is None or off+8>len(z64):raise RuntimeError('ZIP64 compressed size')
        cs=struct.unpack('<Q',z64[off:off+8])[0]
    if (us,cs)!=(info.file_size,info.compress_size):raise RuntimeError('local/central size')

def digitize(wf_v: np.ndarray) -> np.ndarray:
    """把电压波形(V)量化到 14-bit ADC 再还原成 V(带量化台阶)。"""
    wf_v = np.asarray(wf_v, dtype=np.float64)
    adc = np.round((wf_v - V_MIN) / (V_MAX - V_MIN) * LEVELS)
    adc = np.clip(adc, 0, LEVELS)                       # 防越界削顶
    return (adc / LEVELS) * (V_MAX - V_MIN) + V_MIN

def _normalize_waveform(wf: np.ndarray) -> np.ndarray:
    """基线扣除 + 峰值归一化到 ~[0, 1]。"""
    wf = np.asarray(wf, dtype=np.float32)
    base = wf[: max(1, len(wf) // 5)]  # 前 20% 当基线
    wf = wf - float(np.mean(base))     # 扣基线
    peak = float(np.max(wf))
    if peak > 0:
        wf = wf / peak                 # 峰值缩放到 1
    return wf.astype(np.float32)

def _window_waveform(wf: np.ndarray, tp0: int,
                     seq_len: int, trigger_index: int) -> np.ndarray:
    """以 tp0 为基准、trigger_index 为窗口内触发位置，切出长 seq_len 的窗口。

    越界部分用边缘值填充（mode='edge'），保证输出长度恒为 seq_len。
    """
    if not (0 <= trigger_index < seq_len):
        raise ValueError(f"trigger_index {trigger_index} out of [0, {seq_len})")
    left = int(tp0) - trigger_index    # 窗口左端在原波形里的下标
    right = left + seq_len
    src_left = max(0, left)
    src_right = min(wf.size, right)
    seg = wf[src_left:src_right]
    if seg.size == 0:
        raise ValueError("cannot window an empty waveform segment")
    pad_left = max(0, -left)           # 左侧不够则前补
    pad_right = max(0, right - wf.size)  # 右侧不够则后补
    if pad_left or pad_right:
        seg = np.pad(seg, (pad_left, pad_right), mode="edge")
    if seg.size < seq_len:             # 仍不足则尾补到 seq_len
        seg = np.pad(seg, (0, seq_len - seg.size), mode="edge")
    elif seg.size > seq_len:           # 过长则截断
        seg = seg[:seq_len]
    return np.ascontiguousarray(seg, dtype=np.float32)

def constant_fraction_tp0(wf: np.ndarray, fraction: float = 0.5) -> int:
    """触发下标：主峰相连上升支穿越峰值 fraction 处的样本。

    恒定分数定时（CFD）对 PMT/SiPM 闪烁脉冲比前沿阈值更稳健
    （消除了幅度依赖的走步）。
    """
    wf = np.asarray(wf, dtype=np.float64)
    if wf.size == 0:
        raise ValueError("empty waveform")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    peak = int(np.argmax(wf))
    peak_value = float(wf[peak])
    if not np.isfinite(peak_value) or peak_value <= 0:
        return peak
    thr = fraction * peak_value                  # 阈值 = fraction × 峰值
    # The final below-threshold sample locates the branch connected to the main
    # peak. A preceding isolated spike must not become the pulse's time origin.
    below = np.flatnonzero(wf[:peak] < thr)
    if below.size == 0:
        return peak
    return int(below[-1] + 1)

def stream_member(f,info,want,indices,on_row):
 local_header(f,info);f.reserve('compressed',info.compress_size);f.reserve('inflate',info.file_size+1)
 width=want['row_bytes'];chosen=[0] if want['name']=='time_ns.npy' else sorted(indices)
 ranges=[(128+i*width,128+(i+1)*width,i) for i in chosen]
 header=bytearray();row=bytearray(width);filled=0;cursor=0;total=0;crc=0;pending=b'';remaining=info.compress_size;dec=zlib.decompressobj(-15);header_checked=False;delivered=0
 while True:
  f.check()
  if not pending and remaining:
   n=min(1048576,remaining);pending=f.returned_read('compressed',n)
   if len(pending)!=n:raise RuntimeError('short compressed member')
   remaining-=n
  limit=min(1048576,info.file_size+1-total)
  if limit<=0:raise RuntimeError('inflate beyond declared member')
  try:part=dec.decompress(pending,limit)
  except BaseException:f.cost['inflate_exception_internal']='unknown <= retained reservation';raise
  pending=dec.unconsumed_tail;f.cost['actual_returned']['inflate']+=len(part);start=total;stop=start+len(part);total=stop;crc=zlib.crc32(part,crc)
  if start<128:header.extend(part[:min(len(part),128-start)])
  if len(header)==128 and not header_checked:
   if header[:8]!=b'\x93NUMPY\x01\x00' or struct.unpack('<H',header[8:10])[0]+10!=128:raise RuntimeError('NPY v1 exact128')
   h=ast.literal_eval(header[10:128].decode('latin1').strip())
   if h!={'descr':want['dtype'],'fortran_order':False,'shape':tuple(want['shape'])}:raise RuntimeError('NPY exact schema before any row decode')
   header_checked=True
  while cursor<len(ranges) and ranges[cursor][0]<stop:
   lo,hi,index=ranges[cursor];a=max(lo,start);b=min(hi,stop)
   if a<b:
    if not header_checked or a-lo!=filled:raise RuntimeError('selected row schema/continuity')
    row[filled:filled+b-a]=part[a-start:b-start];filled+=b-a
   if filled==width:on_row(index,bytes(row));delivered+=1;filled=0;cursor+=1
   else:break
  if dec.eof:
   if pending or dec.unused_data or remaining:raise RuntimeError('DEFLATE trailing/unconsumed')
   break
  if not part and not pending and not remaining:raise RuntimeError('DEFLATE missing EOF')
 if total!=info.file_size or crc&0xffffffff!=info.CRC or delivered!=len(chosen) or filled:raise RuntimeError('full CRC/EOF/selected delivery')
 f.cost['members'].append({'name':info.filename,'compressed_bytes':info.compress_size,'uncompressed_bytes':total,'crc32':hex(crc&0xffffffff),'selected_rows':delivered,'selected_bytes':delivered*width,'header_sha256':hashlib.sha256(header).hexdigest(),'unselected_numeric_decodes':0})
 return bytes(header)
def read_raw_source(plan,indices,cost,until,on_time,on_wave):
 spec=plan['container'];p=pathlib.Path(spec['path'])
 if p.is_symlink() or p.stat().st_size!=spec['bytes']:raise RuntimeError('original path/size')
 with p.open('rb') as raw:
  before=os.fstat(raw.fileno());f=Counted(raw,cost,plan,until);f.reserve('hash',spec['bytes']);cost['hash_passes']+=1;h=hashlib.sha256();left=spec['bytes']
  while left:
   n=min(left,1048576);b=f.returned_read('hash',n)
   if len(b)!=n:raise RuntimeError('whole original SHA short')
   h.update(b);left-=n
  if h.hexdigest()!=spec['sha256']:raise RuntimeError('original SHA mismatch; no retry')
  cost['container_sha256']=h.hexdigest()
  with zipfile.ZipFile(f,'r',allowZip64=True) as z:
   infos=z.infolist()
   if [x.filename for x in infos]!=[x['name'] for x in plan['members']]:raise RuntimeError('central ordered members')
   for info,want in zip(infos,plan['members']):
    if (info.file_size,info.compress_size,info.CRC,info.compress_type)!=(want['uncompressed_bytes'],want['compressed_bytes'],int(want['crc32'],16),8) or info.flag_bits not in (0,2048) or info.extract_version>45:raise RuntimeError('central exact/flags')
   wants={x['name']:x for x in plan['members']}
   # Time qualification before physical waveform conversion; no other member open.
   stream_member(f,z.getinfo('time_ns.npy'),wants['time_ns.npy'],[0],on_time)
   stream_member(f,z.getinfo('waveforms.npy'),wants['waveforms.npy'],indices,on_wave)
  after=os.fstat(raw.fileno());fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
  cost['fstat_before']={k:getattr(before,k) for k in fields};cost['fstat_after']={k:getattr(after,k) for k in fields}
  if cost['fstat_before']!=cost['fstat_after']:raise RuntimeError('same-handle original drift')
def allocate_interface():
 return {'block_mean_V_f64':np.full((6800,500),np.nan,dtype='<f8'),'window_normalized_f32':np.full((6800,480),np.nan,dtype='<f4'),'window_V_f32':np.full((6800,480),np.nan,dtype='<f4'),'raw_witness_V_f64':np.full((64,10000),np.nan,dtype='<f8'),'time_ns_f64':np.full(10000,np.nan,dtype='<f8'),'normalization_f32':np.full((6800,2),np.nan,dtype='<f4'),'geometry_i32':np.full((6800,9),-1,dtype='<i4'),'signed_metrics_f64':np.full((6800,8),np.nan,dtype='<f8'),'QC_reason_u16':np.zeros(6800,dtype='<u2'),'DEV_bitwise_u8':np.zeros(2000,dtype='|u1')}
QC={'source_nonfinite':1,'downsample_nonfinite':2,'converter_peak_nonpositive':4,'float32_peak_nonpositive':8,'quantization_clipped':16,'padding':32,'window_nonfinite':64,'signed_total_nonpositive_or_nonfinite':128,'signed_PSD_nonfinite':256}
def convert_row(position,b,a,old_real,witness_slot):
 w=np.frombuffer(b,dtype='<f8');bad=0
 if position in witness_slot:a['raw_witness_V_f64'][witness_slot[position]]=w
 if not np.all(np.isfinite(w)):a['QC_reason_u16'][position]=QC['source_nonfinite'];return
 down=w.astype(np.float64).reshape(500,20).mean(axis=1);a['block_mean_V_f64'][position]=down
 if not np.all(np.isfinite(down)):a['QC_reason_u16'][position]=QC['downsample_nonfinite'];return
 adc=np.round((down-V_MIN)/(V_MAX-V_MIN)*LEVELS);low=int(np.count_nonzero(adc<0));high=int(np.count_nonzero(adc>LEVELS));q64=digitize(down);b64=q64[:100].mean();p64=float((q64-b64).max())
 if low or high:bad|=QC['quantization_clipped']
 if p64<=0:a['QC_reason_u16'][position]=bad|QC['converter_peak_nonpositive'];return
 converter_tp=constant_fraction_tp0((q64-b64)/p64,.5)
 q32=np.ascontiguousarray(q64,dtype=np.float32);base=float(np.mean(q32[:100]));peak=float(np.max(q32-base));a['normalization_f32'][position]=[base,peak]
 if peak<=0:a['QC_reason_u16'][position]=bad|QC['float32_peak_nonpositive'];return
 normalized=_normalize_waveform(q32);tp=constant_fraction_tp0(normalized,.5);left=tp-120;right=left+480;pl=max(0,-left);pr=max(0,right-500)
 norm=_window_waveform(normalized,tp,480,120);volts=_window_waveform(q32,tp,480,120);a['window_normalized_f32'][position]=norm;a['window_V_f32'][position]=volts;a['geometry_i32'][position]=[converter_tp,tp,constant_fraction_tp0(norm,.5),left,right,pl,pr,low,high]
 if old_real is not None and position<2000:a['DEV_bitwise_u8'][position]=int(norm.astype('<f4',copy=False).tobytes()==old_real[position*1920:(position+1)*1920])
 if pl or pr:bad|=QC['padding']
 if not np.all(np.isfinite(norm)) or not np.all(np.isfinite(volts)):a['QC_reason_u16'][position]=bad|QC['window_nonfinite'];return
 v=volts.astype(np.float64);z=norm.astype(np.float64);bv=float(v[:100].mean());bz=float(z[:100].mean());vf=v-bv;zf=z-bz;qf=float(vf[100:150].sum());qt=float(vf[100:400].sum());nf=float(zf[100:150].sum());nt=float(zf[100:400].sum());ps=1-qf/qt if qt>0 and math.isfinite(qt) and math.isfinite(qf) else np.nan;ns=1-nf/nt if nt>0 and math.isfinite(nt) and math.isfinite(nf) else np.nan
 a['signed_metrics_f64'][position]=[qf*1000,qt*1000,ps,nf,nt,ns,bv,float(v[:100].std(ddof=1))]
 if not math.isfinite(qt) or qt<=0:bad|=QC['signed_total_nonpositive_or_nonfinite']
 if not math.isfinite(ps):bad|=QC['signed_PSD_nonfinite']
 a['QC_reason_u16'][position]=bad
