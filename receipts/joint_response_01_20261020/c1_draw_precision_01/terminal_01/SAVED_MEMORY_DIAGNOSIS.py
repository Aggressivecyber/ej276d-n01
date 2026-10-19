from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import torch

root=Path('E:/GAN/research/N01');out=Path(__file__).resolve().parent
script=root/'scripts/diagnose_joint01_c1_draw_precision.py'
observation=root/'src/n01/observation008.py'
# A synthetic chunk tests the indexing/storage operation, not any waveform law.
rows=torch.arange(64,dtype=torch.int64).repeat_interleave(32)
z=torch.arange(len(rows)*8,dtype=torch.float32).reshape(len(rows),8)
noise={h:torch.arange(len(rows)*384,dtype=torch.float64).reshape(len(rows),384) for h in ('fast','slow','common_level')}
bank=dict(rows=rows,z=z,noise=noise)
take=torch.arange(len(rows)).reshape(-1,32)[:,:32].reshape(-1)
old=dict(rows=bank['rows'][take],z=bank['z'][take],noise={h:bank['noise'][h][take] for h in noise})
new=bank
values=[bank['rows'],bank['z'],*bank['noise'].values()]
copies=[old['rows'],old['z'],*old['noise'].values()]
for a,b in zip(values,copies,strict=True):
    assert torch.equal(a,b) and a.untyped_storage().data_ptr()!=b.untyped_storage().data_ptr()
assert new is bank
per_row=8+8*4+3*384*8
one_bank=6465*32*per_row
assert sum(v.numel()*v.element_size() for v in values)==2048*per_row
review=json.loads((out/'TERMINAL_REVIEW.json').read_bytes())
record=dict(status='PASS_SYNTHETIC_STORAGE_IDENTITY_CHECK',UTC=datetime.now(timezone.utc).isoformat(),
    source_files={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (script,observation)},
    authoritative_noise_shape=[384],authoritative_noise_dtype='float64',hypotheses=3,
    one_full_32_draw_bank_tensor_bytes=one_bank,two_banks_tensor_bytes=2*one_bank,
    unnecessary_two_bank_32_draw_copies_bytes=2*one_bank,
    old_high_draw_live_original_and_copy_tensor_bytes=4*one_bank,
    proposed_high_draw_live_tensor_bytes=2*one_bank,
    all_five_synthetic_arrays_exact=True,old_five_storage_copies_distinct=True,new_alias_no_copy=True,
    raw_waveforms_generated=0,parameter_updates=0,
    actual_peak_RSS_bytes=review['peak_RSS_bytes'],
    inference='All-row advanced indexing duplicates both complete 32-draw banks. This is a proven avoidable allocation. The exact retained allocator/other-process-memory contribution to the second-head RSS peak was not measured.',
    proposed_completion='Reuse complete 32-draw banks directly; only reconstruct four-draw subsets for exact replay. Complete only missing seed1 direction and validation with unchanged seeds, precision, gates and 12 GiB guard.',
    native_memory_fix_verified=False,scientific_acceptance=False)
(out/'SAVED_MEMORY_DIAGNOSIS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
