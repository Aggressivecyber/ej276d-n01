from pathlib import Path
import ast
r=Path('E:/GAN/research/N01');w=Path(__file__).parent
s=(r/'receipts/response069_20261018/deployment_01/REGISTER_STARTUP.py').read_text()
s=s.replace('response069','response070').replace('WARP068','LEVEL069').replace('LEVELWARP','SPLITWARP').replace('VOLTAGEWARP','LEVELWARP')
s=s.replace("prior=r/'receipts/response068_20261018/wake_20261018_1949_01/collected/run_01'","prior=r/'receipts/response069_20261018/wake_20261018_2019_01/collected/run_01'")
s=s.replace("k.startswith('COUPLED067_')","k.startswith('WARP068_')")
s=s.replace("rec['geometry']['crossing_level_warp_enabled']==enabled","rec['geometry']['crossing_level_warp_enabled'] is True and rec['geometry']['split_area_warp_enabled']==enabled")
s=s.replace('query_error=0.;amplitude_scale_error=0.;knot_error=0.','query_error=0.;amplitude_scale_error=0.;knot_error=0.;split_mean_error=0.;split_random_error=0.;seam_error=0.')
needle="        queries=arrays[label+'_WARP_QUERIES.npz'];assert np.array_equal(queries['parent_row'],np.arange(6593))"
s=s.replace(needle,needle+'''
        prior_queries=dict(np.load(prior/'seed0_LEVELWARP_WARP_QUERIES.npz'))
        assert set(queries)==set(prior_queries)
        for key in queries:assert np.array_equal(queries[key],prior_queries[key]),key
        assert np.array_equal(rows['reference_split_index'],rows['old_mean_peak_index']+30)
        for side in ('front','tail'):
            original_area=rows['positive_'+side+'_area_Vns'];mapped_area=rows['interpolated_positive_'+side+'_area_Vns']
            assert np.min(original_area)>0. and np.min(mapped_area)>0.
            amplitude_scale_error=max(amplitude_scale_error,float(np.max(np.abs(original_area/mapped_area-rows[side+'_amplitude_scale']))))
            split_mean_error=max(split_mean_error,float(np.max(np.abs(rows['reference_'+side+'_integral_residual_Vns']))))
            split_random_error=max(split_random_error,float(np.max(np.abs(rows['random_reference_'+side+'_integral_residual_Vns']))))
        assert np.max(np.abs(rows['positive_front_area_Vns']+rows['positive_tail_area_Vns']-rows['positive_area_Vns']))<1e-12
        seam_error=max(seam_error,float(np.max(np.abs(rows['new_split_step_V']-rows['original_split_step_V']-rows['correction_split_step_V']))))
        assert max(split_mean_error,split_random_error,seam_error)<1e-12
''')
a=s.index('        query=expected_queries[row];vp=');b=s.index('        expected_mean_delta=',a)
s=s[:a]+'''        query=expected_queries[row];vp=np.interp(query,np.arange(66),positive)*mask;vr=np.interp(query,np.arange(66),active)*mask
        if enabled:
            front=np.arange(110,176)<peak+30;moved=np.zeros(66);moved_random=np.zeros(66)
            for side,select in [('front',front),('tail',~front)]:
                part_area=math.fsum(positive[select]);part_v=math.fsum(vp[select]);alpha=part_area/part_v
                moved[select]=alpha*vp[select];provisional=alpha*vr[select];leak=math.fsum(provisional)-math.fsum(active[select]);moved_random[select]=provisional-leak*moved[select]/part_area
                amplitude_scale_error=max(amplitude_scale_error,abs(alpha-rows[side+'_amplitude_scale'][row]))
                assert abs(part_area-rows['positive_'+side+'_area_Vns'][row])<1e-12 and abs(part_v-rows['interpolated_positive_'+side+'_area_Vns'][row])<1e-12
                assert abs(leak-rows['random_'+side+'_provisional_leakage_Vns'][row])<1e-12
                split_mean_error=max(split_mean_error,abs(math.fsum((new-old)[110:176][select])))
                split_random_error=max(split_random_error,abs(math.fsum(delta[110:176][select])))
            cut=peak+30
            for key,val in [('original_split_step_V',old[cut]-old[cut-1]),('new_split_step_V',new[cut]-new[cut-1]),('correction_split_step_V',(new-old)[cut]-(new-old)[cut-1])]:seam_error=max(seam_error,abs(float(val-rows[key][row])))
            assert max(split_mean_error,split_random_error,seam_error)<1e-12
        else:
            alpha=area/math.fsum(vp);moved=alpha*vp;provisional=alpha*vr;moved_random=provisional-(math.fsum(provisional)-math.fsum(active))*moved/area
            amplitude_scale_error=max(amplitude_scale_error,abs(alpha-rows['amplitude_scale'][row]))
'''+s[b:]
s=s.replace('CORRECTLY_STARTED_CROSSING_LEVEL_WARP_AND_068_REPLAY_VERIFIED','CORRECTLY_STARTED_SPLIT_AREA_WARP_AND_069_REPLAY_VERIFIED')
s=s.replace('original068_baseline_arrays_exact','original069_baseline_arrays_exact').replace('knot_max_error_samples=knot_error,','knot_max_error_samples=knot_error,split_mean_integral_max_error_Vns=split_mean_error,split_random_integral_max_error_Vns=split_random_error,split_seam_metadata_max_error_V=seam_error,queries_and_knots_exact069=True,')
s=s.replace('original M0/0686H','original M0/0696H').replace('original068 baseline','original069 baseline').replace('Crossing-level voltage warp correctly started','Split-area crossing-level warp correctly started')
s=s.replace("len(d['jobs'])==88","len(d['jobs'])==89").replace('response070_crossing_level_voltage_warp','response070_split_area_crossing_level_warp')
s=s.replace('all6593 queries and seven-knot pairs,78 mean/random examples','all6593 queries and seven-knot pairs,split-integral invariants,78 mean/random examples')
s=s.replace('Both-head/final guards not yet collected.','Split integral residuals and seam metadata checked for all rows; both-head/final guards not yet collected.')
ast.parse(s);(w/'STARTUP_VERIFIER_SOURCE.py').write_text(s)
print('070 independent split-area startup verifier prepared')
