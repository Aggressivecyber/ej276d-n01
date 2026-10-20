from pathlib import Path
import ast,json

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
prior=root/'receipts/joint_response_01_20261020/c1_objective_alignment_01'
registration=json.loads((here/'REGISTRATION.json').read_bytes())
old_config_sha='578a1145aaae8574c302c69093628ab71d7b8bf7b320e71c648494e0c55bbb92'
replacements={
 'joint_response01_c1_objective_alignment_20261020':'joint_response01_original_draw_precision_20261020',
 'joint_response_01_c1_objective_alignment_20261020':'joint_response_01_original_draw_precision_20261020',
 'diagnose_joint01_c1_objective_alignment':'diagnose_joint01_original_draw_precision',
 'joint-response01-c1-objective-alignment':'joint-response01-original-draw-precision',
 old_config_sha:registration['config_sha256'],
 '796':str(registration['source_files_expected']),
 'Original-teacher precision':'Original-generator draw precision',
}
for filename in ('PREPARE_DEPLOYMENT.py','REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','REMOTE_WAIT_ESTIMATE_REQUEST.py','COLLECT.py','REMOTE_COLLECT.py'):
    source=(prior/filename).read_text(encoding='utf-8')
    for old,new in replacements.items():source=source.replace(old,new)
    if filename=='PREPARE_DEPLOYMENT.py':
        source=source.replace('c1_original_precision_01/SOURCE_MANIFEST.json','c1_objective_alignment_01/SOURCE_MANIFEST.json')
        source=source.replace("previous=base/'joint_response01_c1_original_precision_20261020'","previous=base/'joint_response01_c1_objective_alignment_20261020'")
    if filename=='REMOTE_DISPATCH_REQUEST.py':
        source=source.replace('gpu_free<3840','gpu_free<6144').replace('Less than3.75GiB free GPU for2.75GiB allocator plus1GiB headroom','Less than6GiB free GPU for4GiB allocator plus2GiB headroom')
        source=source.replace("'420s'","'5700s'")
    if filename=='REMOTE_COLLECT.py':
        source=source.replace('32*1024**2','128*1024**2').replace('c1_objective_alignment_terminal_01.tar','original_draw_precision_terminal_01.tar')
    ast.parse(source)
    (here/filename).write_text(source,encoding='utf-8',newline='\n')
print('Deployment, bounded dispatch, status and collection requests prepared.')
