"""Adapt mature standard-library artificial identity tests, no execution here."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
old=(ROOT/'receipts/wp3_015_20261015/readback_engineering_01/check_readback_control.py').read_text('utf8')
text=old.replace('wp3_015_20261015','wp3_016_20261015').replace('readback015','readback016').replace('CONFIG015_SHA','CONFIG016_SHA').replace('MANIFEST015_SHA','MANIFEST016_SHA')
text=text.replace("repo/'WP3_SOURCE_MANIFEST.json'","repo/'WP3_016_SOURCE_MANIFEST.json'").replace("repo/'wp3_015_config.json'","repo/'wp3_016_config.json'")
text=text.replace("len(frozen['files']) == 105","len(frozen['files']) == 116").replace('frozen105','frozen116').replace('all_72','all_48').replace('original300','original211')
text=text.replace('len(reports) == 72','len(reports) == 48').replace('range(14)','range(6)').replace('range(35)','range(23)').replace('== 285','== 200').replace('finite015','finite016').replace('Audit_285','Audit_200')
text=text.replace("['bank_pairs_verified'] = 35","['bank_pairs_verified'] = 23")
anchor="    def fake_pair():"
insert="""    def fake_charge():
        audit.op('SYNTHETIC_CHARGE_NO_ARRAY',dict(charge_npz_files_decoded=1,charge_A_rows_verified=26372,
            charge_A_float64_cells_verified=26372*384,charge_mu_draws_verified=26372,charge_q_draws_verified=26372,
            charge_parent_moment_records_verified=6593),lambda:None)
    for _ in range(2):fake_charge()

"""
assert text.count(anchor)==1;text=text.replace(anchor,insert+anchor,1)
anchor="    require(audit.entered == audit.completed == 200"
assert text.count(anchor)==1;text=text.replace(anchor,'    for _ in range(6):fake_charge()\n'+anchor,1)
text=text.replace('original105_plus2_controls_unchanged=True','original116_plus2_controls_unchanged=True')
with (HERE/'check_readback_control016.py').open('x',encoding='utf8') as f:f.write(text)
print('local stdlib checker created')
