"""Assembly fragment: inserted into016 worker, not imported at runtime."""


def charge_repeat_fields(values):
    """Independent blockwise actual-float64 diagnostics, never replace a draw."""
    x=np.asarray(values)
    require(x.dtype==np.float64 and x.ndim in (2,3) and x.shape[:2]==(P,R) and np.isfinite(x).all(), 'charge actual repeat schema')
    if x.ndim==2:x=x[:,:,None]
    shape=(P,R);result={k:np.zeros(shape,dtype=dtype) for k,dtype in
        [('absmax',np.float64),('absmax_coordinate',np.int64),('relative_max_nonzero_reference',np.float64),
         ('ordered_float64_ULP_max',np.uint64),('ULP_max_coordinate',np.int64),('nonzero_cells',np.int64),('nonzero_with_zero_reference',np.int64)]}
    square_sum=0.;bit_cells=0;changed_parents=[];sign=np.uint64(1<<63)
    for start in range(0,P,128):
        stop=min(start+128,P);block=x[start:stop];ref=block[:,:1];delta=block-ref;absolute=np.abs(delta)
        relative=np.divide(absolute,np.abs(ref),out=np.zeros_like(absolute),where=ref!=0)
        bits,rbits=block.view(np.uint64),ref.view(np.uint64)
        left=np.where(bits&sign,~bits,bits^sign);right=np.where(rbits&sign,~rbits,rbits^sign)
        ulp=np.maximum(left,right)-np.minimum(left,right)
        result['absmax'][start:stop]=absolute.max(2);result['absmax_coordinate'][start:stop]=absolute.argmax(2)
        result['relative_max_nonzero_reference'][start:stop]=relative.max(2)
        result['ordered_float64_ULP_max'][start:stop]=ulp.max(2);result['ULP_max_coordinate'][start:stop]=ulp.argmax(2)
        result['nonzero_cells'][start:stop]=(delta!=0).sum(2)
        result['nonzero_with_zero_reference'][start:stop]=((ref==0)&(delta!=0)).sum(2)
        square_sum+=float(np.sum(delta**2,dtype=np.float64));bit_cells+=int((bits!=rbits).sum())
        changed_parents.extend((np.flatnonzero((delta!=0).any((1,2)))+start).tolist())
    return result,dict(maximum=float(result['absmax'].max()),RMS=math.sqrt(square_sum/x.size),
        nonzero_cells=int(result['nonzero_cells'].sum()),nonzero_rows=int((result['nonzero_cells']>0).sum()),
        changed_parent_rows=changed_parents,bit_cells=bit_cells)


def verify_charge_arrays(arrays,diagnostic,semantic,*,bank,equal,parent_ids=None,input_hashes=None):
    """Saved rows only: no SOURCE archive, random generator or model access."""
    def exact(value,shape,dtype,name):
        require(value.shape==shape and value.dtype==dtype,name+':shape/dtype')
    A,mu,q=arrays['A_signal_pre384_V'],arrays['Q_mu_draws_Vns'],arrays['q_sig_draws_Vns']
    exact(A,(N,384),np.float64,'actual A');exact(mu,(P,R),np.float64,'actual mu');exact(q,(P,R),np.float64,'actual q')
    require(np.isfinite(A).all() and (A[:,:100]==0).all(),'finite frozen A/prefix100')
    require(np.isfinite(mu).all() and np.isfinite(q).all() and (mu>0).all() and (q>0).all(),'strict positive actual mu/q; no floor')
    require(np.array_equal(arrays['source_parent_row'],np.arange(P))
        and np.array_equal(arrays['A_source_parent_row'],np.repeat(np.arange(P),R))
        and np.array_equal(arrays['draw_index'],np.tile(np.arange(R),P)),'charge parent/draw order')
    require(arrays['source_parent_row'].dtype.kind in 'iu' and arrays['A_source_parent_row'].dtype.kind in 'iu'
        and arrays['draw_index'].dtype.kind in 'iu','charge integer lineage')
    ids=arrays['source_parent_ids'];require(ids.shape==(P,) and len(np.unique(ids))==P,'charge unique parent IDs')
    if parent_ids is not None:require(np.array_equal(ids,parent_ids),'same charge/hard SOURCE identity')
    hashes=[]
    for key in ('source_input_sha256_uint8','forward_float64_input_sha256_uint8'):
        value=arrays[key];exact(value,(N,32),np.uint8,key)
        grid=value.reshape(P,R,32);require(np.all(grid==grid[:,:1]),key+':saved per-parent repeats')
        hashes.append(grid[:,0].copy())
    if input_hashes is not None:
        require(all(np.array_equal(a,b) for a,b in zip(hashes,input_hashes)),'saved input digests differ across banks/snapshots')
    require(semantic['bank']==diagnostic['bank']==bank and diagnostic['parent_count']==P
        and diagnostic['actual_rows']==N and diagnostic['draws_per_parent']==R,'charge JSON identity')
    require(diagnostic['numerical_tolerance_V']==1e-7 and diagnostic['tolerance_source']=='same_frozen_parameter_baseline_replay_1e-7_V',
        'unchanged013 point gate identity')
    a,b=float(semantic['a_Vns']),float(semantic['b']);require(math.isfinite(a) and math.isfinite(b) and a>0 and b>0,'positive saved coefficients')
    v=a*mu+b*mu**2;residual=q-mu;sample_variance=q.var(1,ddof=1);mu_variance=mu.var(1,ddof=1)
    require(np.isfinite(v).all() and (v>0).all(),'finite positive per-draw theory variance')
    expected=dict(Q_mu_Vns=mu.mean(1),sample_mean_Vns=q.mean(1),sample_variance_Vns2=sample_variance,
        conditional_mean_MCSE_Vns=np.sqrt(sample_variance/R),sample_mean_MCSE_Vns=np.sqrt(sample_variance/R),
        theory_conditional_variance_Vns2=v.mean(1),theory_conditional_variance_draws_Vns2=v,
        theory_expected_sample_variance_Vns2=v.mean(1)+mu_variance,theory_sample_mean_MCSE_Vns=np.sqrt(v.sum(1))/R,
        Q_mu_numeric_sample_variance_Vns2=mu_variance,Q_mu_numeric_range_Vns=np.ptp(mu,axis=1),
        draw_centered_residuals_Vns=residual,draw_centered_residual_mean_Vns=residual.mean(1),
        draw_centered_residual_second_moment_Vns2=np.square(residual).mean(1),
        draw_centered_residual_sample_variance_Vns2=residual.var(1,ddof=1),
        draw_centered_sample_mean_MCSE_Vns=np.sqrt(residual.var(1,ddof=1)/R))
    for key,value in expected.items():
        require(arrays[key].dtype==np.float64,key+':float64')
        equal(value,arrays[key],'charge:'+key,atol=0.,rtol=128*np.finfo(np.float64).eps)
    Afields,Acheck=charge_repeat_fields(A.reshape(P,R,384));mufields,mucheck=charge_repeat_fields(mu)
    for prefix,fields in [('A_numeric_repeat_',Afields),('Q_mu_numeric_repeat_',mufields)]:
        for key,value in fields.items():equal(value,arrays[prefix+key],'charge:'+prefix+key,atol=0.,rtol=128*np.finfo(np.float64).eps)
    require(Acheck['maximum']<=1e-7 and diagnostic['within_numerical_tolerance'] is True,'actual A exceeds fixed0131e-7V gate')
    for key,value in [('actual_max_abs_residual_V',Acheck['maximum']),('actual_RMS_residual_V',Acheck['RMS']),
        ('nonzero_cells',Acheck['nonzero_cells']),('nonzero_rows',Acheck['nonzero_rows']),
        ('different_float64_bit_cells',Acheck['bit_cells']),('affected_source_parent_rows',np.asarray(Acheck['changed_parent_rows'],np.int64))]:
        equal(value,diagnostic[key],'charge JSON:'+key,atol=0.,rtol=128*np.finfo(np.float64).eps)
    require(np.array_equal(ids[Acheck['changed_parent_rows']],diagnostic['affected_source_ids']),'changed A parent IDs')
    u=np.finfo(np.float64).eps/2;gamma=255*u/(1-255*u);canonical=A[:,110:366]
    cpu_mu=canonical.sum(1,dtype=np.float64).reshape(P,R)
    bound=np.nextafter(2*gamma/(1-gamma)*np.abs(canonical).sum(1).reshape(P,R)+512*np.finfo(np.float64).smallest_subnormal,np.inf)
    difference=mu-cpu_mu
    require(np.all(np.abs(difference)<=bound),'canonical CPU/GPU reduction bound')
    saved_cpu=arrays['Q_mu_CPU_recomputed_draws_Vns'];saved_difference=arrays['Q_mu_GPU_minus_CPU_draws_Vns']
    exact(saved_cpu,(P,R),np.float64,'saved CPU canonical sum');exact(saved_difference,(P,R),np.float64,'saved canonical residual')
    require(np.isfinite(saved_cpu).all() and np.isfinite(saved_difference).all(),'finite saved canonical diagnostics')
    # Both CPU reductions act on the same saved A. A cancellation residual is
    # not a scale for a relative-only comparison between two legal reductions.
    require(np.all(np.abs(saved_cpu-cpu_mu)<=bound) and np.all(np.abs(saved_difference)<=bound),'saved CPU sum within derived reduction bound')
    require(np.array_equal((mu-saved_cpu).view(np.uint64),saved_difference.view(np.uint64)),'charge saved canonical residual algebra bits')
    for key,value in [('Q_mu_CPU_GPU_reduction_bound_Vns',bound),('Q_mu_repeat_bound_from_A_gate_Vns',256e-7+bound+bound[:,:1])]:
        equal(value,arrays[key],'charge:'+key,atol=0.,rtol=64*np.finfo(np.float64).eps)
    require(diagnostic['Q_mu_CPU_GPU_within_derived_bound'] is True,'producer canonical bound marker')
    for name,value in [('Q_mu_CPU_GPU_absmax_Vns',float(np.abs(saved_difference).max()))]:
        equal(value,diagnostic[name],'charge JSON:'+name,atol=0.,rtol=128*np.finfo(np.float64).eps)
    # This producer scalar used GPU/chunk reductions. Retain mature generic
    # engineering tolerance for an equivalent saved-A CPU reduction.
    expected_increment_RMS=math.sqrt(float(np.sum(A*A,axis=1).dot((a/mu+b).reshape(-1))/(N*384)))
    equal(expected_increment_RMS,semantic['actual_A_expected_increment_RMS_V'],'charge expected increment RMS JSON')
    for name,value in [('conditional_CV_min',float(np.sqrt(a/mu+b).min())),('conditional_CV_mean',float(np.sqrt(a/mu+b).mean())),
        ('conditional_CV_max',float(np.sqrt(a/mu+b).max())),('finite_MC_mean_error_RMS_Vns',float(np.sqrt(np.square(residual.mean(1)).mean()))),
        ('finite_MC_mean_MCSE_RMS_Vns',float(np.sqrt(np.square(expected['sample_mean_MCSE_Vns']).mean())))]:
        equal(value,semantic[name],'charge semantic JSON:'+name)
    for name,state in arrays.items():
        if name.endswith('_state'):require(state.dtype==np.uint8 and state.ndim==1 and len(state)>0,'saved RNG byte state schema')
    return ids.copy(),hashes,dict(bank=bank,parents=P,draws=R,A_rows=N,A_samples=384,
        actual_A_repeat_max_abs_V=Acheck['maximum'],actual_A_repeat_RMS_V=Acheck['RMS'],actual_mu_repeat_max_abs_Vns=mucheck['maximum'],
        Q_CPU_GPU_absmax_Vns=float(np.abs(difference).max()),Q_CPU_GPU_bound_max_Vns=float(bound.max()),
        saved_vs_independent_CPU_sum_absmax_Vns=float(np.abs(saved_cpu-cpu_mu).max()),
        saved_canonical_residual_algebra_verified=True,
        actual_mu_min_Vns=float(mu.min()),actual_mu_max_Vns=float(mu.max()),q_min_Vns=float(q.min()),q_max_Vns=float(q.max()),
        mean_error_RMS_Vns=float(np.sqrt(np.square(residual.mean(1)).mean())),
        conditional_random_variance_mean_Vns2=float(v.mean()),expected_sample_variance_mean_Vns2=float((v.mean(1)+mu_variance).mean()),
        independent_expected_A_increment_RMS_V=expected_increment_RMS,
        numerical_mu_sample_variance_mean_Vns2=float(mu_variance.mean()),numerical_mu_spread_is_physical_variance=False,
        independent_random_mean_theory_MCSE_RMS_Vns=float(np.sqrt(np.square(expected['theory_sample_mean_MCSE_Vns']).mean())),
        draw_centered_second_moment_mean_Vns2=float(np.square(residual).mean()),
        actual_A_sha256=hashlib.sha256(A.tobytes()).hexdigest(),actual_mu_sha256=hashlib.sha256(mu.tobytes()).hexdigest(),
        actual_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),theory_v_sha256=hashlib.sha256(v.tobytes()).hexdigest(),
        saved_input_hashes_verified_only_for_internal_repeats_and_cross_bank_identity=True,original_SOURCE_rehashed=False,
        q_latent_or_RNG_replay=False,charge_r8_derived=False,full_stochastic_preADC_saved=False,
        finite_draw_moments_are_not_exact_analytic_sample_realizations=True,
        mean_or_first_draw_substitution=False,independent_frozen_A_saved_array_readback=True)
