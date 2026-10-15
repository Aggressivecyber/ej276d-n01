"""Build the local016 delta over completed015 sources; no runtime or remote action."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OLD=ROOT/'receipts/wp3_015_20261015';NEW=HERE.parent;OUT=NEW/'postrun_independent_01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def once(text,old,new):
    assert text.count(old)==1,(text.count(old),old[:80])
    return text.replace(old,new,1)
def write(path,data):
    with path.open('xb') as f:f.write(data if isinstance(data,bytes) else data.encode('utf8'))

plan=read(OUT/'READBACK_PLAN.json');assert sha(OUT/'READBACK_PLAN.json')=='9fd6ed29156c5567ef9738835e00fb06f3c2a214bb5694a267b22f06b34e938e'
assert all(v==0 for v in plan['scientific_budget'].values()) and plan['expected_operations']==200
wp=OLD/'postrun_independent_01/readback_saved_arrays.py';cp=OLD/'readback_engineering_01/readback_control.py';kp=OLD/'postrun_independent_01/saved_array_kernels.py'
assert sha(wp)=='c2f53542b2bdc2557cfc6f1470f11113e4fc8d2be0ec13648e154d05222272a7'
assert sha(cp)=='85f19797e91ec62951a4bfc2eb54b46a05fc550a3d28e4ded9346e077477b66d'
assert sha(kp)=='fe1366f35f36e97974f1a4037a36ad858c03dd8453957c1b45b65080f0fee3ac'
old=wp.read_text('utf8');text=old.replace('Bounded015','Bounded016').replace('wp3_015_saved_readback_20261015','wp3_016_saved_readback_20261015')
text=text.replace('wp3_015_config.json','wp3_016_config.json').replace('WP3_SOURCE_MANIFEST.json','WP3_016_SOURCE_MANIFEST.json')
for oldsha,newsha in [('54c096ad1a5e112493798e5c6c8db481cd3c79b78747200af0bbd47bccba5394',plan['frozen_configuration']['sha256']),
    ('f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a',plan['frozen_manifest']['sha256']),
    ('9d253a57bb79d9f25b112b4f5e0009a239c21360766c89cef7357d040adebe42',plan['inventory']['sha256'])]:text=once(text,oldsha,newsha)
text=once(text,"TRIALS = ('seed0_M0', 'seed0_M1', 'seed1_M0', 'seed1_M1')","TRIALS = ('seed0_Q1', 'seed1_Q1')")
text=text.replace("('initial_evaluation', ('A', 'initial'))","('initial_evaluation', ('A', 'M0', 'initial'))")
text=text.replace('all72 hard files','all48 hard files').replace('frozen015','frozen016').replace('registered015','registered016')
text=text.replace("len(mf['files']) == 105, 'frozen105source'","len(mf['files']) == 116, 'frozen116source'")
text=once(text,'np = kernels = None','EXPECTED_COUNTS = '+repr(plan['expected_complete_counters'])+'\nnp = kernels = None')
fragment=(HERE/'charge_checks_fragment.py').read_text('utf8');fragment=fragment[fragment.index('def charge_repeat_fields'):]
text=once(text,'class Audit:',fragment+'\n\nclass Audit:')
text=once(text,'self.touched, self.baseline_hashes, self.raw_hashes = {}, {}, {}',
    'self.touched, self.baseline_hashes, self.raw_hashes = {}, {}, {}\n        self.charge_banks, self.saved_input_hashes = [], None')
text=once(text,"'probe_snapshots_verified')","'probe_snapshots_verified', 'charge_npz_files_decoded', 'charge_A_rows_verified', 'charge_A_float64_cells_verified',\n                 'charge_mu_draws_verified', 'charge_q_draws_verified', 'charge_parent_moment_records_verified')")
method='''    def charge_bank(self, trial, snapshot, bank):
        directory=self.root/'runs/primary/artifacts'/trial/snapshot
        path=directory/f'bank{bank}_CHARGE_PREADC_MOMENTS.npz'
        diagnostic=load(directory/f'bank{bank}_A_NUMERIC_REPEATS.json')
        require(diagnostic['saved_arrays_sha256']==self.touched[path.relative_to(self.root).as_posix()]['sha256'],'charge diagnostic binds raw SHA')
        semantics=load(directory/'CHARGE_SEMANTICS.json')
        selected=[s for s in semantics['banks'] if s['bank']==bank];require(len(selected)==1,'unique charge bank semantic JSON')
        with np.load(path,allow_pickle=False) as z:
            arrays={name:z[name] for name in z.files}
        parents,hashes,summary=verify_charge_arrays(arrays,diagnostic,selected[0],bank=bank,equal=self.equal,
            parent_ids=self.parents,input_hashes=self.saved_input_hashes)
        if self.parents is None:self.parents=parents
        if self.saved_input_hashes is None:self.saved_input_hashes=hashes
        self.charge_banks.append(dict(summary,trial=trial,snapshot=snapshot,
            saved_source_input_digests_sha256=hashlib.sha256(hashes[0].tobytes()).hexdigest(),
            saved_forward_input_digests_sha256=hashlib.sha256(hashes[1].tobytes()).hexdigest()))
        del arrays
        gc.collect()
        self.guard()

'''
text=once(text,'    def one_bank(self, trial, snapshot, arm, hyp, bank, baseline, qfloor, peakfloor):',method+'    def one_bank(self, trial, snapshot, arm, hyp, bank, baseline, qfloor, peakfloor):')
start=text.index("        summary_only = 'runs/primary/artifacts/MEAN_EVIDENCE_REAL_SUMMARY.npz'")
stop=text.index('        for name, row in raw.items():',start)
text=text[:start]+'''        for trial in TRIALS:
            for snapshot in ('initial_evaluation','final_evaluation'):
                expected_evaluation.update('runs/primary/artifacts/'+trial+'/'+snapshot+f'/bank{b}_CHARGE_PREADC_MOMENTS.npz' for b in (0,1))
        require(len(expected_evaluation)==112 and set(raw)==expected_evaluation,'exact016112 raw saved-array scope')
        require(sum(row['bytes'] for row in raw.values())==2169655337 and sum(row['bytes'] for row in raw.values())<=3072*1024**2,
                'finite actual016 raw-byte envelope')
'''+text[stop:]
text=once(text,'                baseline = audit.load_baselines(trial, snapshot)\n                for arm in arms:',
'''                baseline = audit.load_baselines(trial, snapshot)
                for bank in (0,1):
                    audit.op('charge_bank_saved_statistics',dict(charge_npz_files_decoded=1,charge_A_rows_verified=N,
                        charge_A_float64_cells_verified=N*384,charge_mu_draws_verified=N,charge_q_draws_verified=N,
                        charge_parent_moment_records_verified=P),lambda bank=bank:audit.charge_bank(trial,snapshot,bank),
                        trial=trial,snapshot=snapshot,bank=bank)
                for arm in arms:''')
start=text.index("        require(len(audit.pairs) == 36")
stop=text.index('        result = dict(audit.base()',start)
text=text[:start]+'''        require(len(audit.pairs)==24 and len(audit.charge_banks)==8 and not audit.probes,'complete016 finite readback without probes')
        require(not audit.inflight and audit.counts==audit.attempted==EXPECTED_COUNTS,'exact016 complete counters')
        require(audit.entered==audit.completed==200,'exact016200 operations')
'''+text[stop:]
text=once(text,'            pairs=audit.pairs, probes=audit.probes,','            pairs=audit.pairs, charge_banks=audit.charge_banks, probes=audit.probes,')
text=once(text,'''            mean_evidence_REAL_summary_SHA_only_no_decode=True,
            mean_evidence_summary_record=audit.touched[summary_only],
            probe_or_gradient_readback_applicable=False,
            inherited015_reference_mean_module_not_causal_identification=True,''',
'''            frozen_A_preADC384_independently_recomputed_from_saved_rows=True,
            full_stochastic_preADC384_carrier_independently_replayed=False,
            original_SOURCE_input_byte_hash_independently_replayed=False,
            saved_input_digest_repetition_and_crossbank_identity_only=True,
            charge_latent_or_RNG_replay=False,charge_r8_added=False,
            probe_or_gradient_readback_applicable=False,
            charge_mean_protection_is_not_physical_response_identification=True,''')
assert 'summary_only' not in text and 'wp3_015' not in text
ast.parse(text);write(OUT/'readback_saved_arrays.py',text);write(OUT/'saved_array_kernels.py',kp.read_bytes())
write(HERE/'WORKER_DELTA.patch',''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='mature015/worker',tofile='new016/worker')))

old=cp.read_text('utf8');text=old.replace("ORIGINAL='/home/rylan/projects/ej276d-n01/wp3_015_20261015'",f"ORIGINAL='{plan['original_root']}'")
text=text.replace('wp3_015_saved_readback_20261015','wp3_016_saved_readback_20261015').replace('CONFIG015_SHA','CONFIG016_SHA').replace('MANIFEST015_SHA','MANIFEST016_SHA')
text=text.replace('wp3_015_config.json','wp3_016_config.json').replace('WP3_SOURCE_MANIFEST.json','WP3_016_SOURCE_MANIFEST.json')
for oldsha,newsha in [('32ee8e020d653491c8d97ae5e41f9aca18098824b11c6f6333b3635cc0d7dd50',sha(OUT/'READBACK_PLAN.json')),
    ('54c096ad1a5e112493798e5c6c8db481cd3c79b78747200af0bbd47bccba5394',plan['frozen_configuration']['sha256']),
    ('f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a',plan['frozen_manifest']['sha256'])]:text=once(text,oldsha,newsha)
text=text.replace('original015','original016').replace('original105','original116').replace('original300/161','original211/112')
text=text.replace("len(manifest['files'])==105","len(manifest['files'])==116").replace("len(inventory['whole_remote_inventory'])==300","len(inventory['whole_remote_inventory'])==211")
text=text.replace("len(inventory['omitted_remote_inventory'])==161","len(inventory['omitted_remote_inventory'])==112").replace("len(reports)==72","len(reports)==48")
text=text.replace("complete.get('all_four_trials_completed') is True and complete['counters']['G_updates']==8000", "complete.get('all_two_trials_completed') is True and complete['counters']['G_updates']==4000")
text=text.replace('saved_REPORT_JSON_verified=72','saved_REPORT_JSON_verified=48')
text=text.replace('raw161 single SHA pass (160 evaluation files decode; mean-evidence summary SHA only)','raw112 single SHA and decode pass')
text=once(text,"    return dict(original116_verified=True,original_native_complete_verified=True,saved_REPORT_JSON_verified=48,",
'''    extra=['runs/primary/artifacts/'+trial+'/'+snapshot+'/'+name for trial in ('seed0_Q1','seed1_Q1')
        for snapshot in ('initial_evaluation','final_evaluation') for name in ('CHARGE_SEMANTICS.json','bank0_A_NUMERIC_REPEATS.json','bank1_A_NUMERIC_REPEATS.json')]
    extra.append('runs/primary/artifacts/CHARGE_INITIALIZATION.json')
    for name in extra:require(record(root/name)==inventory['whole_remote_inventory'][name],'charge metadata changed: '+name)
    return dict(original116_verified=True,original_native_complete_verified=True,saved_charge_JSON_verified=13,saved_REPORT_JSON_verified=48,''')
text=text.replace("c['raw_files_SHA_checked']==161", "c['raw_files_SHA_checked']==112 and c['charge_npz_files_decoded']>=2")
start=text.index('    expected=dict(raw_files_SHA_checked=161,');stop=text.index("    raw=read(Path(SIDE)/'BOUND_ORIGINAL_INVENTORY.json')",start)
text=text[:start]+'    expected='+repr(plan['expected_complete_counters'])+"\n    require(done['counters']==done['attempted_counters']==expected,'incomplete finite saved-array scope')\n    require(done['operations_entered']==done['operations_completed']==200,'readback finite operation count differs')\n"+text[stop:]
text=text.replace('completed_pairs=36','completed_pairs=24')
ast.parse(text);write(HERE/'readback_control.py',text)
write(HERE/'CONTROLLER_DELTA.patch',''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='mature015/controller',tofile='new016/controller')))
print(json.dumps(dict(local_sources_created=True,worker_sha=sha(OUT/'readback_saved_arrays.py'),kernel_sha=sha(OUT/'saved_array_kernels.py'),controller_sha=sha(HERE/'readback_control.py'))))
