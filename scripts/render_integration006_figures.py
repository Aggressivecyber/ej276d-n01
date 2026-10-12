"""Reopen final PDFs and render every page for separate human visual QA."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess

os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1];ID='integration_006_20261012'
REPORT=ROOT/'reports'/ID;OUT=ROOT/'runs'/ID/'visual_QA'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'Preserve earlier QA renders'
    OUT.mkdir()
    render=shutil.which('pdftoppm');assert render,'Bundled Poppler required'
    pdfs=sorted((REPORT/'figures').glob('*.pdf'));assert len(pdfs)==4
    rows=[]
    for pdf in pdfs:
        reader=PdfReader(pdf);assert len(reader.pages)==1
        page=reader.pages[0];text=page.extract_text();assert len(text.strip())>40
        prefix=OUT/pdf.stem
        subprocess.run([render,'-png','-singlefile','-r','110',str(pdf),str(prefix)],check=True)
        png=prefix.with_suffix('.png');assert png.is_file() and png.stat().st_size>1000
        rows.append(dict(pdf=pdf.relative_to(ROOT).as_posix(),pdf_sha256=sha(pdf),pages=1,
            page_width_pt=float(page.mediabox.width),page_height_pt=float(page.mediabox.height),
            text_characters=len(text),rendered_png=png.relative_to(ROOT).as_posix(),rendered_png_sha256=sha(png)))
    receipt=dict(status='all_four_PDFs_reopened_and_rendered_pending_visual_inspection',
        UTC=dt.datetime.now(dt.timezone.utc).isoformat(),renderer=render,files=rows)
    (OUT/'PDF_RENDER_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
