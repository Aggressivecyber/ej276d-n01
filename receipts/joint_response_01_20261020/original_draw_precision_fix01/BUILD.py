from pathlib import Path
import ast

root=Path('E:/GAN/research/N01')
source=(root/'scripts/diagnose_joint01_original_draw_precision.py').read_text(encoding='utf-8')
source=source.replace('from diagnose_joint01_native_direction import native_values','from diagnose_joint01_native_direction import native_values as original_native_values')
source=source.replace("ROOT = Path(__file__).resolve().parents[1]",'''# Bind the exact 8:1 operator in this isolated scientific process only.
# NativeMetricLoss036.wave_components uses its module's imported W1 symbol;
# inherited QPSD/noise methods resolve the losses008_v2 module symbol.
import run_response036 as native_wave_module
from n01 import losses008_v2 as native_distribution_module
from n01.joint01_integer_w1 import empirical_wasserstein_1d
native_wave_module.empirical_wasserstein_1d = empirical_wasserstein_1d
native_distribution_module.empirical_wasserstein_1d = empirical_wasserstein_1d

NATIVE_TIMING = dict(calls=0,seconds=0.)


def native_values(observations,target,setup,counts):
    started=time.monotonic()
    values=original_native_values(observations,target,setup,counts)
    NATIVE_TIMING['calls']+=1
    NATIVE_TIMING['seconds']+=time.monotonic()-started
    return values


ROOT = Path(__file__).resolve().parents[1]''')
source=source.replace("        saved = np.load(ROOT/p['low_draw_arrays'][seed_index],allow_pickle=False)","""        saved = np.load(ROOT/p['low_draw_arrays'][seed_index],allow_pickle=False)
        if seed_index==0:
            predecessor=np.load(ROOT/p['stopped_prefix_arrays'],allow_pickle=False)
            prefix_native_errors=[]""")
source=source.replace("            progress_file(out,counts,'original_precision_bank_ready',seed=seed_index,role=label)","""            if seed_index==0:
                np.testing.assert_array_equal(arrays[label+'_baseline_residual'],predecessor[label+'_baseline_residual'])
                np.testing.assert_allclose(arrays[label+'_baseline_native'],predecessor[label+'_baseline_native'],rtol=1e-12,atol=1e-12)
                prefix_native_errors.append(float(np.max(np.abs(arrays[label+'_baseline_native']-predecessor[label+'_baseline_native']))))
                counts['same_32draw_centers_verified']+=1
            progress_file(out,counts,'original_precision_fix01_bank_ready',seed=seed_index,role=label,native_timing=dict(NATIVE_TIMING))""")
source=source.replace("                report['FD_columns_completed'] = coordinate + 1", """                if seed_index==0 and coordinate+1==p['stopped_prefix_columns']:
                    for key in ('FD_plus_residual','FD_minus_residual','FD_plus_theta','FD_minus_theta'):
                        np.testing.assert_array_equal(arrays[key],predecessor[key])
                    for key in ('FD_plus_native','FD_minus_native'):
                        np.testing.assert_allclose(arrays[key],predecessor[key],rtol=1e-12,atol=1e-12)
                        prefix_native_errors.append(float(np.max(np.abs(arrays[key]-predecessor[key]))))
                    counts['predecessor_FD_columns_verified']+=p['stopped_prefix_columns']
                    report['same_32draw_prefix_replay']=dict(columns=p['stopped_prefix_columns'],centers=5,
                        joint_and_parameter_arrays_exact=True,max_native_absolute_difference=max(prefix_native_errors))
                    predecessor.close()
                report['FD_columns_completed'] = coordinate + 1
                report['native_timing']=dict(NATIVE_TIMING)""")
source=source.replace("progress_file(out, counts, 'original_precision_joint_Jacobian', seed=seed_index, column=coordinate + 1)","progress_file(out, counts, 'original_precision_fix01_joint_Jacobian', seed=seed_index, column=coordinate + 1,native_timing=dict(NATIVE_TIMING))")
source=source.replace('generator_readout_and_metric_code_unchanged=True,','generator_and_readout_code_unchanged=True,native_distance_definition_unchanged=True,native_8_to_1_implementation_changed=True,native_timing=dict(NATIVE_TIMING),')
ast.parse(source)
(root/'scripts/diagnose_joint01_original_draw_precision_fix01.py').write_text(source,encoding='utf-8',newline='\n')
print('Exact-W1 diagnostic repair built; original generator and metric definition retained.')
