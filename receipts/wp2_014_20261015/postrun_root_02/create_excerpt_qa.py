"""Preserve the exact report CSS and later section subtrees for native QA."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
report = root / 'reports/wp2_014_20261015/N01_WP2_014_REPORT.html'
text = report.read_text(encoding='utf-8')
out = Path(__file__).resolve().parent / 'html_qa_02'
out.mkdir(exist_ok=False)
prefix = text[:text.index('<main>')]
prefix = prefix.replace('<style>', '<base href="'+report.parent.as_uri()+'/">'+'<style>', 1)
files = {}
for name, section in (('wave',3),('conditional',5),('readback',9)):
    start = text.index('<h2 id="section%d">' % section)
    end = text.index('<h2 id="section%d">' % (section+1))
    subtree = text[start:end]
    target = out / (name+'.html')
    target.write_text(prefix+'<main>'+subtree+'</main>', encoding='utf-8')
    files[target.name] = dict(section=section, exact_original_subtree_SHA256=hashlib.sha256(subtree.encode()).hexdigest(),
                             preview_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(out/'PREVIEW_BINDINGS.json').write_text(json.dumps(dict(report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
    purpose='Native rendering of exact later section subtrees with original CSS and assets. Original top viewport rendered separately. Headless fragment captures were blank/clipped and are retained as harness failures.',
    original_report_modified=False, files=files), ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status='exact_CSS_section_previews_created', files=len(files), output=str(out))))
