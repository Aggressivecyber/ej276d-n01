from pathlib import Path
from datetime import datetime,timezone
import ctypes,ctypes.wintypes as w,json,os,subprocess,sys,time
h=Path(__file__).resolve().parent
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
class MemoryCounters(ctypes.Structure):
 _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
k=ctypes.WinDLL('kernel32',use_last_error=True);k.SetProcessAffinityMask.argtypes=[w.HANDLE,ctypes.c_size_t];k.SetProcessAffinityMask.restype=w.BOOL
ps=ctypes.WinDLL('psapi',use_last_error=True);ps.GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.POINTER(MemoryCounters),w.DWORD];ps.GetProcessMemoryInfo.restype=w.BOOL
with (h/('seed'+sys.argv[1]+'_STDOUT.txt')).open('x') as stdout,(h/('seed'+sys.argv[1]+'_STDERR.txt')).open('x') as stderr:
 p=subprocess.Popen([sys.executable,'-B',str(h/'AUDIT_HARD_OUTPUTS.py'),sys.argv[1]],cwd=h,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
 if not k.SetProcessAffinityMask(w.HANDLE(int(p._handle)),15):raise ctypes.WinError(ctypes.get_last_error())
 started=time.monotonic();utc=datetime.now(timezone.utc).isoformat();peak=0;reason=None
 while p.poll() is None:
  m=MemoryCounters();m.cb=ctypes.sizeof(m)
  if not ps.GetProcessMemoryInfo(w.HANDLE(int(p._handle)),ctypes.byref(m),m.cb):raise ctypes.WinError(ctypes.get_last_error())
  peak=max(peak,m.PeakWorkingSetSize)
  if peak>1024**3 or time.monotonic()-started>330:
   reason='RSS_or_external_wall_budget_exceeded';p.kill();break
  time.sleep(1)
 code=p.wait()
 receipt=dict(exit_code=code,reason=reason,started_UTC=utc,ended_UTC=datetime.now(timezone.utc).isoformat(),peak_RSS_bytes=peak,elapsed_seconds=time.monotonic()-started,pid=p.pid,CPU_affinity_mask=15,RSS_limit_bytes=1024**3,external_wall_seconds=330)
 (h/('seed'+sys.argv[1]+'_EXECUTION.json')).write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
 sys.exit(code)
