"""Render original016 HTML and exact sections with Edge and E-drive caches."""
import datetime as dt
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REPORT=ROOT/'reports/wp3_016_20261015/REPORT.html'
EDGE=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
parser=argparse.ArgumentParser()
parser.add_argument('--out-name',default='html_qa_01')
args=parser.parse_args()
assert args.out_name in ['html_qa_01','html_qa_02']
OUT=HERE/args.out_name
CACHE=Path('E:/GAN/runtime_tmp/wp3_016_edge_qa')/args.out_name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

assert EDGE.is_file() and REPORT.is_file()
manifest=json.loads((REPORT.parent/'report_manifest.json').read_bytes())
assert sha(REPORT)==manifest['outputs']['REPORT.html']['sha256']
original_sha=sha(REPORT)
OUT.mkdir(exist_ok=False);CACHE.mkdir(parents=True,exist_ok=True)
original=REPORT.read_text(encoding='utf-8')
prefix=original[:original.index('<main>')]
prefix=prefix.replace('<style>','<base href="'+REPORT.parent.as_uri()+'/"><style>',1)
jobs=[('top',REPORT,1200)];bindings={}
for name,section,height in [('wave',1,2400),('absolute_T',3,2400),('readback',5,2400)]:
    begin=original.index('<h2 id="section%d">'%section)
    end=original.index('<h2 id="section%d">'%(section+1))
    subtree=original[begin:end]
    path=OUT/(name+'.html');path.write_text(prefix+'<main>'+subtree+'</main>',encoding='utf-8',newline='\n')
    bindings[name]=dict(section=section,exact_subtree_sha256=hashlib.sha256(subtree.encode()).hexdigest(),
        HTML_sha256=sha(path),original_CSS_and_assets_preserved=True)
    jobs.append((name,path,height))
env=os.environ.copy();env.update(TEMP=str(CACHE),TMP=str(CACHE),LOCALAPPDATA=str(CACHE/'localappdata'))
renders={}
for name,source,height in jobs:
    png=OUT/(name+'.png')
    cmd=[str(EDGE),'--headless=new','--disable-gpu','--no-first-run','--disable-extensions',
        '--allow-file-access-from-files','--hide-scrollbars','--user-data-dir='+str(CACHE/('profile_'+name)),
        '--disk-cache-dir='+str(CACHE/('cache_'+name)),'--window-size=1440,%d'%height,
        '--screenshot='+str(png),source.as_uri()]
    with (OUT/(name+'_edge.log')).open('wb') as log:
        process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=35,
            creationflags=subprocess.CREATE_NO_WINDOW)
    data=png.read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n'
    width,actual_height=struct.unpack('>II',data[16:24]);assert width==1440 and actual_height==height
    renders[name]=dict(source=str(source),source_sha256=sha(source),png=str(png),png_sha256=sha(png),png_bytes=len(data),
        width=width,height=height,Edge_native_exit=process.returncode)
    print(json.dumps(dict(render=name,Edge_native_exit=process.returncode,png_bytes=len(data))),flush=True)
assert sha(REPORT)==original_sha
receipt=dict(schema='n01/wp3-016-report-native-render/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    report_sha256=original_sha,original_report_modified=False,scope='Original top viewport plus exact wave/absoluteT/readback subtrees with original CSS/assets; representative native views.',
    visual_inspection_pending=True,bindings=bindings,renders=renders,new_scientific_operations=0)
(OUT/'RENDER_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
