"""Render original HTML and exact later section subtrees using the installed Edge."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/wp3_015_20261015/REPORT.html'
EDGE = Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
OUT = HERE / 'html_qa_01'
CACHE = Path('E:/GAN/runtime_tmp/wp3_015_edge_qa')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert EDGE.is_file() and REPORT.is_file()
    OUT.mkdir(exist_ok=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    original = REPORT.read_text(encoding='utf-8')
    prefix = original[:original.index('<main>')]
    prefix = prefix.replace('<style>', '<base href="' + REPORT.parent.as_uri() + '/"><style>', 1)
    jobs = [('top', REPORT, 1200)]
    bindings = {}
    for name, section, height in [('wave', 3, 2400), ('conditional', 5, 1700), ('readback', 9, 1700)]:
        begin = original.index('<h2 id="section%d">' % section)
        end = original.index('<h2 id="section%d">' % (section+1))
        subtree = original[begin:end]
        path = OUT / (name+'.html')
        path.write_text(prefix+'<main>'+subtree+'</main>', encoding='utf-8')
        bindings[name] = dict(section=section, exact_subtree_sha256=hashlib.sha256(subtree.encode()).hexdigest(),
            HTML_sha256=digest(path), original_CSS_and_assets_preserved=True)
        jobs.append((name, path, height))
    env = os.environ.copy()
    env.update(TEMP=str(CACHE), TMP=str(CACHE), LOCALAPPDATA=str(CACHE/'localappdata'))
    results = {}
    for name, source, height in jobs:
        png = OUT / (name+'.png')
        cmd = [str(EDGE), '--headless=new', '--disable-gpu', '--no-first-run', '--disable-extensions',
            '--allow-file-access-from-files', '--hide-scrollbars', '--user-data-dir='+str(CACHE/('profile_'+name)),
            '--disk-cache-dir='+str(CACHE/('cache_'+name)), '--window-size=1440,%d'%height,
            '--screenshot='+str(png), source.as_uri()]
        with (OUT/(name+'_edge.log')).open('wb') as log:
            process = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=35,
                creationflags=subprocess.CREATE_NO_WINDOW)
        data = png.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        width, actual_height = struct.unpack('>II', data[16:24])
        assert width == 1440 and actual_height == height
        results[name] = dict(source=str(source), source_sha256=digest(source), png=str(png), png_sha256=digest(png),
            png_bytes=len(data), width=width, height=height, Edge_native_exit=process.returncode)
    assert digest(REPORT) == '7d46bfcecb8726f3be029e1f335abf5d7b4826cae0e643612cbe998b5f29aeb6'
    receipt = dict(schema='n01/wp3-report-native-render/v1', UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        report_sha256=digest(REPORT), original_report_modified=False,
        scope='Original top viewport and exact later section subtrees with original CSS/assets; representative native renders.',
        visual_inspection_pending=True, bindings=bindings, renders=results, new_scientific_operations=0)
    (OUT/'RENDER_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='four_native_renders_written_visual_inspection_pending', report_sha256=digest(REPORT),
        native_exits={k:v['Edge_native_exit'] for k,v in results.items()})))


if __name__ == '__main__':
    main()
