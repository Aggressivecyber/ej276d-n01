from pathlib import Path
import hashlib,json,os,subprocess
root=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_diagnostic_20261020')
m=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
for name,entry in m['files'].items():
 raw=(root/name).read_bytes()
 assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+':'+str(root/'scripts'),TMPDIR=str(root/'cache'),XDG_CACHE_HOME=str(root/'cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
r=subprocess.run(['/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python','-B','-c','import diagnose_joint01_recentered_derivatives,torch,numpy,scipy,json; print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,CUDA=torch.cuda.is_available())))'],cwd=root,env=env,capture_output=True,text=True)
(root/'IMPORT_STDOUT.txt').write_text(r.stdout);(root/'IMPORT_STDERR.txt').write_text(r.stderr);r.check_returncode()
runtime=json.loads(r.stdout);assert runtime['CUDA']
result=dict(status='PASS_NAS_DEPLOYMENT',files_verified=len(m['files']),source_bytes=sum(v['bytes'] for v in m['files'].values()),config_sha256=m['config_sha256'],runtime=runtime,remote_root=str(root),no_compute_disk_payload_copy=True)
(root/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
