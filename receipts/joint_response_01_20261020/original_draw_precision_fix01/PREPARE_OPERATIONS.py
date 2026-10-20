from pathlib import Path
import ast,json

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
previous=root/'receipts/joint_response_01_20261020/original_draw_precision_01'
registration=json.loads((here/'REGISTRATION.json').read_bytes())
old=json.loads((previous/'REGISTRATION.json').read_bytes())
for filename in ('PREPARE_DEPLOYMENT.py','REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','REMOTE_WAIT_ESTIMATE_REQUEST.py','COLLECT.py','REMOTE_COLLECT.py'):
    source=(previous/filename).read_text(encoding='utf-8')
    for a,b in {
        'joint_response01_original_draw_precision_20261020':'joint_response01_original_draw_precision_fix01_20261020',
        'joint_response_01_original_draw_precision_20261020':'joint_response_01_original_draw_precision_fix01_20261020',
        'diagnose_joint01_original_draw_precision':'diagnose_joint01_original_draw_precision_fix01',
        'joint-response01-original-draw-precision':'joint-response01-original-draw-precision-fix01',
        old['config_sha256']:registration['config_sha256'],
        '805':'815',
        'original_draw_precision_terminal_01.tar':'original_draw_precision_fix01_terminal_01.tar',
    }.items():source=source.replace(a,b)
    if filename=='PREPARE_DEPLOYMENT.py':
        source=source.replace('c1_objective_alignment_01/SOURCE_MANIFEST.json','original_draw_precision_01/SOURCE_MANIFEST.json')
        source=source.replace("previous=base/'joint_response01_c1_objective_alignment_20261020'","previous=base/'joint_response01_original_draw_precision_20261020'")
        source=source.replace("p['method_contract']):","p['method_contract'], 'src/n01/joint01_integer_w1.py'):")
    ast.parse(source);(here/filename).write_text(source,encoding='utf-8',newline='\n')
source=(previous/'AUDIT_COMPLETE.py').read_text(encoding='utf-8')
source=source.replace('joint_response_01_original_draw_precision_20261020.json','joint_response_01_original_draw_precision_fix01_20261020.json')
source=source.replace('==805','==815')
source=source.replace("inputs=json.loads((BASE/'INPUT_MAP.json').read_bytes())", "inputs=json.loads((ROOT/'receipts/joint_response_01_20261020/original_draw_precision_01/INPUT_MAP.json').read_bytes())\ninputs.update(json.loads((BASE/'INPUT_MAP.json').read_bytes()))")
source=source.replace("    low=np.load(Path(inputs[p['low_draw_arrays'][seed]]),allow_pickle=False)","""    low=np.load(Path(inputs[p['low_draw_arrays'][seed]]),allow_pickle=False)
    if seed==0:
        prior=np.load(Path(inputs[p['stopped_prefix_arrays']]),allow_pickle=False)
        for label in p['role_seed_bases']:
            np.testing.assert_array_equal(a[label+'_baseline_residual'],prior[label+'_baseline_residual'])
            np.testing.assert_allclose(a[label+'_baseline_native'],prior[label+'_baseline_native'],rtol=1e-12,atol=1e-12)
        for key in ('FD_plus_residual','FD_minus_residual','FD_plus_theta','FD_minus_theta'):
            np.testing.assert_array_equal(a[key][:,:9],prior[key])
        for key in ('FD_plus_native','FD_minus_native'):
            np.testing.assert_allclose(a[key][:,:9],prior[key],rtol=1e-12,atol=1e-12)
        assert report['same_32draw_prefix_replay']['columns']==9 and report['same_32draw_prefix_replay']['centers']==5
        counts['same_32draw_centers_reverified']+=5
        counts['same_32draw_FD_columns_reverified']+=9
""")
source=source.replace("assert c['frozen_low_candidate_checks']==4", "assert c['frozen_low_candidate_checks']==4\nassert c['same_32draw_centers_verified']==counts['same_32draw_centers_reverified']==5\nassert c['predecessor_FD_columns_verified']==counts['same_32draw_FD_columns_reverified']==9\nassert result['native_timing']['calls']==10+high_calls")
source=source.replace('PASS_SAVED_ORIGINAL_DRAW_PRECISION_AUDIT','PASS_SAVED_ORIGINAL_DRAW_PRECISION_FIX01_AUDIT')
ast.parse(source);(here/'AUDIT_COMPLETE.py').write_text(source,encoding='utf-8',newline='\n')
print('Exact-W1 repair deployment, control and complete audit prepared.')
