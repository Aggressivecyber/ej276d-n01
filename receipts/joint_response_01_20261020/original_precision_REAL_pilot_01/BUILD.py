from pathlib import Path
import ast, difflib

root=Path('E:/GAN/research/N01'); here=Path(__file__).resolve().parent
original=(root/'scripts/run_joint_response01_multibank_real.py').read_text('utf-8')
source=original.replace('Rotating three-bank joint REAL calibration with matched physical controls.',
    'Three-iteration 32-draw joint REAL pilot with matched physical controls.')
source=source.replace('from diagnose_joint01_native_direction import native_values\nfrom diagnose_joint01_secant_trust import hard_check',
    'from diagnose_joint01_original_draw_precision_fix01 import native_values, hard_check, prediction, NATIVE_TIMING')
source=source.replace('progress_file, random_bank, prediction,','progress_file, random_bank,')
source=source.replace("            banks[part] = random_bank(setup, parents, draws, bank_seed, counts)",
'''            bank = random_bank(setup, parents, draws, bank_seed, counts)
            banks[part] = dict(rows=bank['rows'].cpu(), z=bank['z'].cpu(),
                noise={h: bank['noise'][h].cpu() for h in HYPOTHESES})
            arrays[part + '_random_seed'] = np.array(bank_seed, dtype=np.int64)
            del bank''')
source=source.replace("            baseline[part] = prediction(model, theta, banks[part], setup['loss'].q_floor, counts)",
    "            current = prediction(model, theta, banks[part], setup['loss'].q_floor, counts)")
source=source.replace('feature_means(baseline[part], maps[part])','feature_means(current, maps[part])')
source=source.replace('native_values(baseline[part], targets[part], role_setup[part], counts)',
    'native_values(current, targets[part], role_setup[part], counts)')
source=source.replace("            anchor = prediction(model, initial, banks[part], setup['loss'].q_floor, counts)",
'''            baseline[part] = {h: dict(carrier=current[h]['carrier_input'].cpu(),
                flags={key: int(value.sum()) for key, value in current[h]['flags'].items()}) for h in HYPOTHESES}
            del current
            anchor = prediction(model, initial, banks[part], setup['loss'].q_floor, counts)''')
source=source.replace("            arrays[part + '_initial_native'] = anchor_native[part].cpu().numpy()\n        del anchor",
    "            arrays[part + '_initial_native'] = anchor_native[part].cpu().numpy()\n            del anchor")
source=source.replace("                    bn.append(native_values(obs, targets[part], role_setup[part], counts))",
    "                    bn.append(native_values(obs, targets[part], role_setup[part], counts))\n                    del obs")
source=source.replace("        arrays['endpoint'] = theta.cpu().numpy().copy()",
    "        arrays['endpoint'] = theta.cpu().numpy().copy()\n        report['native_timing'] = dict(NATIVE_TIMING)")
source=source.replace("            accepted_updates=sum(item['accepted'] for item in trace))",
    "            accepted_updates=sum(item['accepted'] for item in trace), native_timing=dict(NATIVE_TIMING))\n        del banks, baseline")
source=source.replace("column=coordinate + 1)","column=coordinate + 1, native_timing=dict(NATIVE_TIMING))")
source=source.replace("kernel_controls=kernels, role_counts=", "native_timing=dict(NATIVE_TIMING), kernel_controls=kernels, role_counts=")
source=source.replace('Registered multibank REAL','Registered original precision REAL')
source=source.replace('Pinned multibank REAL','Pinned original precision REAL')
ast.parse(source)
path=root/'scripts/run_joint_response01_original_precision_real.py'
path.write_text(source,encoding='utf-8',newline='\n')
(here/'DRIVER_DIFF.txt').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),
    fromfile='run_joint_response01_multibank_real.py',tofile=path.name)),encoding='utf-8')
print('Built bounded high-draw driver; original loop, solver and final decisions retained.')
