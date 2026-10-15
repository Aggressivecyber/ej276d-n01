"""Record direct visual review of the four final generated report PNGs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT=Path('E:/GAN/research/N01')
HERE=Path(__file__).resolve().parent
FIG=ROOT/'reports/wp3_015_20261015/figures'
checks={
    '01_wave_selection':'Six-cell bar labels, frozen 2% line and 12-bank scores are readable; M0 6/6 and M1 0/6 retained.',
    '02_absolute_tolerance':'All15 calibrated components and12 cells visible. Near-threshold PSD ratios use five decimals, including1.00005; absolute failures remain.',
    '03_mean_residual_gate':'Six saved residual/required RMS pairs and legends visible; finite-population scope explicit.',
    '04_repeated_parent_moments':'Both legends sit above the bars with clear labels; derived4+4 spread and derived8 MCSE distinguished.',
}
result=dict(schema='n01/wp3-015-report-figure-visual-QA/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,
    method='All four PNGs viewed directly; updated figures02 and04 reopened after precision and legend changes.',
    figures={name:dict(sha256=hashlib.sha256((FIG/(name+'.png')).read_bytes()).hexdigest(),observation=note) for name,note in checks.items()},
    HTML_rendering='pending root representative browser QA',scientific_operations=0)
with (HERE/'FIGURE_VISUAL_QA.json').open('x',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
    f.write('\n')
print(json.dumps(dict(passed=True,receipt=str(HERE/'FIGURE_VISUAL_QA.json'))))
