"""Record root's completed inspection of four plots and representative native views."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import struct

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REPORT=ROOT/'reports/wp3_016_20261015'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref=lambda p:dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))

def png(path):
    data=path.read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n'
    width,height=struct.unpack('>II',data[16:24])
    return dict(**ref(path),width=width,height=height)

manifest=json.loads((REPORT/'report_manifest.json').read_bytes())
render_path=HERE/'html_qa_02/RENDER_RECEIPT.json'
renders=json.loads(render_path.read_bytes())
assert renders['report_sha256']==sha(REPORT/'REPORT.html')
for row in renders['renders'].values():assert row['Edge_native_exit']==0 and sha(Path(row['png']))==row['png_sha256']
for name in ['absolute_T','readback']:
    assert (HERE/'html_qa_01'/(name+'.png')).read_bytes()==(HERE/'html_qa_02'/(name+'.png')).read_bytes()
value=dict(schema='n01/wp3-016-root-report-visual-QA/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    passed=True,status='PASS_FOUR_FIGURES_AND_REPRESENTATIVE_NATIVE_VIEWS',report=ref(REPORT/'REPORT.html'),
    native_render_receipt=ref(render_path),original_HTML_top_viewport=png(HERE/'html_qa_02/top.png'),
    exact_section_views={name:png(HERE/'html_qa_02'/(name+'.png')) for name in ['wave','absolute_T','readback']},
    original_English_figures={Path(name).stem:png(REPORT/name) for name in manifest['outputs'] if name.startswith('figures/') and name.endswith('.png')},
    observations=[
        'Viewed all four original English PNGs: labels, units, legends, six-cell comparisons and positive/negative tiny changes are clear.',
        'The absolute-T heatmap retains seven failed components and the near-threshold PSD values; exact ratios also appear in the HTML table and text.',
        'Charge parameters, expected increment RMS and theoretical versus empirical MCSE are distinct and readable; no physical parameter identification is claimed.',
        'Native4 and derived8 precision plus finite-support counts are readable; denominator7 and empirical-rank versus theory scope are explicit.',
        'Viewed original top and exact wave/absoluteT/readback sections with original CSS/assets. Chinese text, tables and images fit with no visible overflow or missing glyphs.',
        'After the denominator wording correction, top/wave were rendered and inspected again; final absoluteT/readback PNGs are byte-identical to the previously inspected views.',
        'The summary explicitly distinguishes12 original distances,6 two-bank means and the final equal-weight mean; the small gain and scientific non-acceptance remain visible.'
    ],scope='Representative native viewports and exact selected section subtrees; all four original plots inspected. This is not full interactive scrolling.',
    original_report_modified_by_visual_QA=False,new_scientific_CP_NN_AD_G_RNG=0)
with (HERE/'REPORT_VISUAL_QA.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(status=value['status'],report_sha256=sha(REPORT/'REPORT.html'),figures=4,native_views=4)))
