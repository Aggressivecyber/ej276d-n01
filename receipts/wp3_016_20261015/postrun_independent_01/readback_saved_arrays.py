"""Bounded016 saved-array verification. Never import neural modules or load CP.

Prepared locally. Root must register its SHA and authorize the one CPU execution.
Native wave/QPSD/noise distances are producer evidence plus JSON algebra; they
are NOT replayed by this script. All recomputation below is stored-row statistics.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

CONFIG_SHA = '5279597aed74feae87448bfc71aa1387bd711609591eabde775142fa6d1303ab'
MANIFEST_SHA = '824780d297c440d0f1bc51b3ea3344e488ec42ccb46af959d9d0412d8240fc14'
INVENTORY_SHA = 'e7cf8c67cdcb2dcadf006b3a8c060f706f9a1b64990c2e383288d79ce22097af'
KERNEL_SHA = 'fe1366f35f36e97974f1a4037a36ad858c03dd8453957c1b45b65080f0fee3ac'
P, R, N = 6593, 4, 26372
H = ('fast', 'slow', 'common_level')
TRIALS = ('seed0_Q1', 'seed1_Q1')
SCALARS = ('Q_Vns', 'tail_Vns', 'PSD_raw', 'PSD_guarded', 'peak_voltage_V',
           'baseline_mean_V', 'baseline_RMS_V', 'rise10_90_ns', 'FWHM_ns')
EXPECTED_COUNTS = {'raw_files_SHA_checked': 112, 'raw_bytes_SHA_read': 2169655337, 'hard_rows_files_decoded': 48, 'report_npz_files_decoded': 48, 'baseline_npz_files_decoded': 8, 'probe_raw_npz_files_decoded': 0, 'probe_gradient_npz_files_decoded': 0, 'waveform_rows_decoded': 1265856, 'bank_pairs_verified': 24, 'probe_snapshots_verified': 0, 'charge_npz_files_decoded': 8, 'charge_A_rows_verified': 210976, 'charge_A_float64_cells_verified': 81014784, 'charge_mu_draws_verified': 210976, 'charge_q_draws_verified': 210976, 'charge_parent_moment_records_verified': 52744}
np = kernels = None


def decode_window(codes, mean, xor=None):
    require(codes.dtype == np.uint16 and codes.ndim == 2 and codes.shape[1] == 256
            and int(codes.max()) <= 16383, 'ADC code dtype/range')
    require(mean.dtype == np.float64 and mean.shape == (len(codes),), 'exact double mean schema')
    value = codes.astype(np.float64)/16383-.5-mean[:, None]
    correction = 0.
    if xor is not None:
        before = value.copy()
        require(xor.dtype == np.uint64 and xor.shape == value.shape, 'IEEE XOR schema')
        np.bitwise_xor(value.view(np.uint64), xor, out=value.view(np.uint64))
        correction = float(np.max(np.abs(value-before)))
    require(np.isfinite(value).all() and correction <= 4e-15, 'decoded finite/013 IEEE arithmetic tolerance')
    require(np.array_equal(np.rint((value+mean[:, None]+.5)*16383).astype(np.uint16), codes), 'exact ADC inverse code roundtrip')
    return value, correction


def lineage(parent, draw, bank_values, ids, *, parents, draws, bank):
    count = parents*draws
    require(parent.dtype.kind in 'iu' and draw.dtype.kind in 'iu', 'integer parent/draw identity')
    require(np.array_equal(parent, np.repeat(np.arange(parents), draws))
            and np.array_equal(draw, np.tile(np.arange(draws), parents)), 'exact original parent/draw order')
    require(ids.shape == (count,) and len(np.unique(ids)) == parents
            and np.all(ids.reshape(parents, draws) == ids.reshape(parents, draws)[:, :1]), 'unique SOURCE IDs repeated per parent')
    require(bank_values.dtype.kind in 'iu' and bank_values.shape == (count,)
            and np.array_equal(bank_values, np.full(count, bank)), 'integer bank identity')
    return ids.reshape(parents, draws)[:, 0].copy()


def wave_moments(values):
    require(values.ndim == 3 and values.shape[1] > 1 and np.isfinite(values).all(), 'finite repeated wave tensor')
    mean = values.mean(1)
    factor = values-mean[:, None]
    return mean, (factor**2).sum(1)/(values.shape[1]-1), factor


def require(value, message):
    if not value:
        raise ValueError(message)


def load(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    result = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def write_new(p, value):
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


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


class Audit:
    def __init__(self, args):
        self.args = args
        self.root, self.out = args.run_root.resolve(), args.out.resolve()
        require(self.root.is_dir() and not self.out.exists() and not self.out.is_relative_to(self.root),
                'fresh sidecar output outside original run tree required')
        self.out.mkdir(parents=True)
        self.started = time.monotonic()
        self.errors, self.pairs, self.probes, self.parents = {}, [], [], None
        self.touched, self.baseline_hashes, self.raw_hashes = {}, {}, {}
        self.charge_banks, self.saved_input_hashes = [], None
        names = ('raw_files_SHA_checked', 'raw_bytes_SHA_read', 'hard_rows_files_decoded',
                 'report_npz_files_decoded', 'baseline_npz_files_decoded', 'probe_raw_npz_files_decoded',
                 'probe_gradient_npz_files_decoded', 'waveform_rows_decoded', 'bank_pairs_verified',
                 'probe_snapshots_verified', 'charge_npz_files_decoded', 'charge_A_rows_verified', 'charge_A_float64_cells_verified',
                 'charge_mu_draws_verified', 'charge_q_draws_verified', 'charge_parent_moment_records_verified')
        self.counts = dict.fromkeys(names, 0)
        self.attempted, self.inflight = self.counts.copy(), {}
        self.entered = self.completed = 0
        self.phase = 'initializing_readonly_audit'
        self.state()

    def base(self):
        return dict(schema='n01/wp3-readback-state/v1', run_id='wp3_016_saved_readback_20261015',
            UTC=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), phase=self.phase,
            counters=self.counts.copy(), attempted_counters=self.attempted.copy(),
            operations_entered=self.entered, operations_completed=self.completed,
            inflight=self.inflight.copy(), unknown_inflight_completion='never_asserted_completed',
            artificial_only=bool(getattr(self.args, 'artificial_only', False)),
            completed_pair_count=self.counts['bank_pairs_verified'],
            scientific_CP_NN_AD_training=0, config_sha256=CONFIG_SHA, source_manifest_sha256=MANIFEST_SHA,
            readback_script_sha256=sha(Path(__file__)), pure_kernel_sha256=KERNEL_SHA,
            inventory_sha256=getattr(self, 'inventory_sha', None))

    def state(self, extra=None):
        p = self.out/'status.json'
        temporary = self.out/'status.json.tmp'
        with temporary.open('w', encoding='utf8', newline='\n') as f:
            json.dump(dict(self.base(), **(extra or {})), f, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(p)

    def journal(self, row):
        with (self.out/'operations.jsonl').open('a', encoding='utf8', newline='\n') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())

    def op(self, name, charges, function, **detail):
        operation_id = self.entered
        self.entered += 1
        self.phase = name
        require(all(k in self.counts and type(v) is int and v >= 0 for k, v in charges.items()), 'audit counter schema')
        for key, value in charges.items():
            self.attempted[key] += value
        row = dict(operation_id=operation_id, event='entered', name=name, charges=charges,
                   UTC=datetime.now(timezone.utc).isoformat(), details=detail)
        self.inflight[str(operation_id)] = row
        self.journal(row)
        self.state()
        try:
            value = function()
        except BaseException as error:
            self.journal(dict(row, event='failed', error_type=type(error).__name__, error=str(error),
                              actual_completion='unknown', UTC=datetime.now(timezone.utc).isoformat()))
            self.state()
            raise
        for key, count in charges.items():
            self.counts[key] += count
        self.completed += 1
        self.inflight.pop(str(operation_id))
        self.journal(dict(row, event='completed', UTC=datetime.now(timezone.utc).isoformat()))
        self.state()
        return value

    def guard(self):
        import resource
        require(time.monotonic() - self.started <= 1800, 'audit wall1800s cap')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        require(rss <= 1536 * 1024**2, 'audit RSS1536MiB cap')
        size = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        require(size <= 64 * 1024**2, 'audit output64MiB cap')
        return dict(wall_seconds=time.monotonic()-self.started, peak_RSS_bytes=rss,
                    output_bytes=size, CPU_threads=2)

    def equal(self, a, b, key, *, atol=2e-13, rtol=2e-11):
        a, b = np.asarray(a), np.asarray(b)
        require(a.shape == b.shape, key + ':shape')
        if a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
            require(np.array_equal(a, b), key + ':integer/flag')
            return
        require(np.array_equal(np.isnan(a), np.isnan(b)), key + ':NaN locations')
        finite = np.isfinite(a) & np.isfinite(b)
        error = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
        require(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), key + ':numeric ' + str(error))
        self.errors[key] = max(self.errors.get(key, 0.), error)

    def member(self, archive, entry, cache):
        key = entry['array']
        if key not in cache:
            cache[key] = archive[key]
        value = cache[key]
        require(list(value.shape) == entry['shape'] and str(value.dtype) == entry['dtype'], 'REPORT array schema:' + key)
        return value

    def identities(self, archive, bank, report_archive=None, report=None, cache=None):
        parent, draw = archive['source_parent_row'], archive['draw_index']
        ids = archive['source_parent_ids']
        parent_ids = lineage(parent, draw, archive['bank_index'], ids, parents=P, draws=R, bank=bank)
        if self.parents is None:
            self.parents = parent_ids
        require(np.array_equal(ids, self.parents[parent]), 'same SOURCE population across all48 hard files')
        if report is not None:
            rp = report['repeated_parent']
            require(rp['parent_count'] == P and rp['draws_per_parent'] == 4
                    and rp['covariance_denominator'] == 3 and rp['covariance_rank_upper_bound'] == 3,
                    'native r4 covariance contract')
            for key, value in [('parent_rows', np.arange(P)), ('bank_grid', np.full((P, 4), bank)),
                               ('draw_grid', np.tile(np.arange(4), (P, 1)))]:
                self.equal(value, self.member(report_archive, rp[key], cache), 'native r4 ' + key)
            require(np.array_equal(self.member(report_archive, report['source_input_trends']['source_ids'], cache), self.parents),
                    'independently saved report SOURCE IDs')

    def load_baselines(self, trial, snapshot):
        result = []
        for bank in (0, 1):
            path = self.root / 'runs/primary/artifacts' / trial / snapshot / f'bank{bank}_BASELINE100.npz'
            def read_baseline(path=path):
                with np.load(path, allow_pickle=False) as z:
                    b = z['baseline100']
                    require(b.shape == (N, 100) and b.dtype == np.float64 and np.isfinite(b).all(), 'actual prefix100 schema')
                    require(np.array_equal(z['source_parent_row'], np.repeat(np.arange(P), 4))
                            and np.array_equal(z['draw_index'], np.tile(np.arange(4), P)), 'baseline parent/draw identity')
                return b
            b = self.op('baseline100_decode', dict(baseline_npz_files_decoded=1), read_baseline,
                        trial=trial, snapshot=snapshot, bank=bank)
            self.baseline_hashes[f'{trial}/{snapshot}/bank{bank}'] = hashlib.sha256(b.tobytes()).hexdigest()
            result.append(b)
        return result

    def charge_bank(self, trial, snapshot, bank):
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

    def one_bank(self, trial, snapshot, arm, hyp, bank, baseline, qfloor, peakfloor):
        stem = f'bank{bank}_{arm}_{hyp}'
        directory = self.root / 'runs/primary/artifacts' / trial / snapshot
        report = load(directory / (stem + '_REPORT.json'))
        require(report['population']['fake_rows'] == N and report['population']['real_rows'] == 21047
                and report['population']['all_population_rows_retained'] is True, 'full population dimensions')
        with np.load(directory / (stem + '_HARD_ROWS.npz'), allow_pickle=False) as z, \
             np.load(directory / (stem + '_REPORT.npz'), allow_pickle=False) as rz:
            cache = {}
            self.identities(z, bank, rz, report, cache)
            codes, mean, peaks = z['ADC_crop_codes'], z['baseline_mean_V'], z['peak_index']
            require(codes.dtype == np.uint16 and codes.shape == (N, 256) and int(codes.max()) <= 16383, 'ADC code dtype/range')
            require(mean.dtype == np.float64 and mean.shape == peaks.shape == (N,)
                    and peaks.dtype.kind in 'iu' and np.all((peaks >= 112) & (peaks < 144)), 'mean/peak schema')
            y, correction = decode_window(codes, mean, z['IEEE754_reconstruction_XOR_uint64']
                                          if 'IEEE754_reconstruction_XOR_uint64' in z.files else None)
            require(np.array_equal(mean.view(np.uint64), z['diagnostic__baseline_mean_V'].view(np.uint64)), 'stored exact native baseline mean')
            self.equal(mean, baseline.mean(1), stem + ':prefix mean', atol=1e-16)
            q, tail = y.sum(1), y[:, 46:].sum(1)
            psd = np.divide(tail, q, out=np.full_like(q, np.nan), where=q != 0)
            centered = baseline-baseline.mean(1, keepdims=True)
            morph = kernels.morphology(y)
            features = dict(Q_Vns=q, tail_Vns=tail, front_Vns=q-tail, PSD_raw=psd,
                PSD_guarded=tail/(np.where(q < 0, -1., 1.)*np.maximum(np.abs(q), qfloor)),
                baseline_mean_V=baseline.mean(1), baseline_RMS_V=np.sqrt(np.mean(centered**2, 1)), **morph)
            for name, value in features.items():
                if name in report['feature_rows']:
                    self.equal(value, self.member(rz, report['feature_rows'][name], cache), stem + ':feature:' + name, atol=2e-12)
            flags = dict(Q_nonpositive=q <= 0, PSD_undefined=q == 0, PSD_guarded=np.abs(q) < qfloor,
                shape_peak_tied=(y == y.max(1, keepdims=True)).sum(1) > 1,
                shape_peak_floor_active=y.max(1) < peakfloor, shape_peak_floor_equal=y.max(1) == peakfloor,
                Q_exact_zero=q == 0, Q_negative=q < 0, Q_log_floor_active=q < qfloor,
                Q_abs_floor_equal=np.abs(q) == qfloor,
                peak_at_search_boundary=(peaks == 112) | (peaks == 143),
                **{name: value for name, value in morph.items() if np.asarray(value).dtype.kind == 'b'})
            for name, value in flags.items():
                raw_name = 'flag__' + name
                if raw_name in z.files:
                    self.equal(value, z[raw_name], stem + ':raw flag:' + name)
                if name in report['observation_flags']:
                    self.equal(value, self.member(rz, report['observation_flags'][name], cache), stem + ':report flag:' + name)
            # Full-carrier/preADC flags cannot be reconstructed; verify preserved
            # row values/counts without pretending that crop256 contains carrier384.
            for name, entry in report['observation_flags'].items():
                saved = self.member(rz, entry, cache)
                require(saved.shape == (N,) and saved.dtype.kind == 'b', 'saved flag schema')
                if 'flag__' + name in z.files:
                    self.equal(saved, z['flag__' + name], stem + ':preserved flag:' + name)
                rate = report['population']['fake_flags'][name]
                require(rate['count'] == int(saved.sum()) and rate['denominator'] == N, 'native saved flag count:' + name)
            scalar = np.stack([features[name] for name in SCALARS], 1).reshape(P, 4, len(SCALARS))
            rp = report['repeated_parent']
            require(rp['scalar_names'] == list(SCALARS), 'nine scalar order')
            x = y.reshape(P, 4, 256)
            wm, wv, unused_factor = wave_moments(x)
            del unused_factor
            self.equal(wm, self.member(rz, rp['hard_wave_mean_V'], cache), stem + ':r4 mean')
            self.equal(wv, self.member(rz, rp['hard_wave_covariance_diagonal_V2'], cache), stem + ':r4 covariance diagonal')
            require(rp['hard_wave_mean_MCSE_V']['reconstruction']
                    == 'sqrt(hard_wave_covariance_diagonal_V2/draws_per_parent)', 'native wave MCSE4 reconstruction rule')
            wave_mcse = np.sqrt(wv/4)
            r4 = dict(waveform_spread_RMS_V=float(np.sqrt(wv.mean())),
                      waveform_MCSE_RMS_V=float(np.sqrt(np.mean(wave_mcse**2))),
                      waveform_MCSE_array_sha256=hashlib.sha256(wave_mcse.tobytes()).hexdigest())
            for name, value in kernels.scalar_moments(scalar).items():
                self.equal(value, self.member(rz, rp['scalars'][name], cache), stem + ':r4 scalar:' + name, atol=2e-11)
            code_hash = hashlib.sha256(np.ascontiguousarray(y).view(np.uint64).tobytes()).hexdigest()
            self.raw_hashes[f'{trial}/{snapshot}/{stem}'] = code_hash
            del cache, centered, morph, codes, mean, peaks
        self.guard()
        return y, scalar, wm, wv, correction, r4

    def pair(self, trial, snapshot, arm, hyp, baseline, qfloor, peakfloor):
        banks = [self.op('hard_bank_saved_statistics',
                 dict(hard_rows_files_decoded=1, report_npz_files_decoded=1, waveform_rows_decoded=N),
                 lambda b=b: self.one_bank(trial, snapshot, arm, hyp, b, baseline[b], qfloor, peakfloor),
                 trial=trial, snapshot=snapshot, arm=arm, hypothesis=hyp, bank=b) for b in (0, 1)]
        var_sum = mcse_sum = 0.
        factor_sha = hashlib.sha256()
        for start in range(0, P, 64):
            stop = min(start+64, P)
            x = np.concatenate([b[0].reshape(P, 4, 256)[start:stop] for b in banks], 1)
            mu, var, factor = wave_moments(x)
            merged_mu = (banks[0][2][start:stop]+banks[1][2][start:stop])/2
            merged_var = (3*(banks[0][3][start:stop]+banks[1][3][start:stop])
                          +2*(banks[0][2][start:stop]-banks[1][2][start:stop])**2)/7
            self.equal(mu, merged_mu, 'r8 merge mean')
            self.equal(var, merged_var, 'r8 merge covariance diagonal')
            factor_sha.update(np.ascontiguousarray(factor).tobytes())
            var_sum += float(var.sum())
            mcse_sum += float((var/8).sum())
            self.guard()
        scalar = np.concatenate([b[1] for b in banks], 1)
        moments = kernels.scalar_moments(scalar)
        # Full9x9 all-pairs/finite supports. The alternate calculation is a
        # literal per-parent centered matrix product for complete cases;
        # pairwise-finite outputs are always retained separately.
        complete = np.isfinite(scalar).all((1, 2))
        if complete.any():
            centered = scalar[complete]-scalar[complete].mean(1, keepdims=True)
            covariance = np.einsum('pri,prj->pij', centered, centered)/7
            self.equal(covariance, moments['sample_covariance'][complete], 'r8 full9x9 complete-case covariance', atol=2e-11)
        self.pairs.append(dict(trial=trial, snapshot=snapshot, arm=arm, hypothesis=hyp,
            parents=P, rows_per_bank=N, r4_native_moments_verified=True, r8_derived_from_saved4plus4=True,
            r8_native_report_exists=False, covariance_denominator=7, waveform_rank_upper_bound=7,
            full_scalar_covariance_shape=[P, 9, 9], pairwise_finite_joint_counts_verified=True,
            complete_case_scalar_parents=int(complete.sum()),
            r4_by_bank=[b[5] for b in banks],
            hard_wave_sqrt_mean_sample_variance_V=math.sqrt(var_sum/(P*256)),
            hard_wave_RMS_conditional_mean_MCSE_V=math.sqrt(mcse_sum/(P*256)),
            centered_wave_factor_sha256=factor_sha.hexdigest(),
            r8_scalar_covariance_sha256=hashlib.sha256(moments['sample_covariance'].tobytes()).hexdigest(),
            r8_scalar_MCSE_sha256=hashlib.sha256(moments['MonteCarlo_SE'].tobytes()).hexdigest(),
            IEEE_correction_absmax_V=max(b[4] for b in banks),
            dense256x256_covariance_materialized=False))
        del banks, scalar, moments
        gc.collect()
        print(json.dumps(dict(phase='pair_verified', pair=len(self.pairs), trial=trial, arm=arm, hypothesis=hyp)), flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    require(os.name == 'posix', 'registered native CPU readback only; no local raw execution')
    audit = Audit(args)
    try:
        import resource
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        resource.setrlimit(resource.RLIMIT_AS, (3072*1024**2, 3072*1024**2))
        if hasattr(os, 'sched_getaffinity'):
            available = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, available[:2])
        def deadline(signum, frame):
            raise TimeoutError('fixed readback wall1800s cap')
        signal.signal(signal.SIGALRM, deadline)
        def interrupted(signum, frame):
            raise InterruptedError('owned readback received signal ' + str(signum))
        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
        signal.setitimer(signal.ITIMER_REAL, 1800)
        root = audit.root
        require(sha(root/'wp3_016_config.json') == CONFIG_SHA and sha(root/'WP3_016_SOURCE_MANIFEST.json') == MANIFEST_SHA, 'frozen016 controls')
        cfg, mf = load(root/'wp3_016_config.json'), load(root/'WP3_016_SOURCE_MANIFEST.json')
        audit.cfg = cfg
        require(len(mf['files']) == 116, 'frozen116source')
        for name, row in mf['files'].items():
            path = root/name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'frozen sourceSHA:' + name)
        require(sha(args.inventory) == INVENTORY_SHA, 'registered016 full inventory SHA')
        inv = load(args.inventory)
        audit.inventory_sha = sha(args.inventory)
        require(inv['config_sha256'] == CONFIG_SHA and inv['source_manifest_sha256'] == MANIFEST_SHA
                and inv['whole_remote_inventory_verified'] is True, 'bound complete remote inventory')
        raw = inv['omitted_remote_inventory']
        expected_evaluation = set()
        for trial in TRIALS:
            for snapshot, arms in [('initial_evaluation', ('A', 'M0', 'initial')), ('final_evaluation', ('final',))]:
                directory = 'runs/primary/artifacts/' + trial + '/' + snapshot + '/'
                expected_evaluation.update(directory + f'bank{b}_BASELINE100.npz' for b in (0, 1))
                expected_evaluation.update(directory + f'bank{b}_{arm}_{h}_{kind}.npz'
                    for b in (0, 1) for arm in arms for h in H for kind in ('HARD_ROWS', 'REPORT'))
        for trial in TRIALS:
            for snapshot in ('initial_evaluation','final_evaluation'):
                expected_evaluation.update('runs/primary/artifacts/'+trial+'/'+snapshot+f'/bank{b}_CHARGE_PREADC_MOMENTS.npz' for b in (0,1))
        require(len(expected_evaluation)==112 and set(raw)==expected_evaluation,'exact016112 raw saved-array scope')
        require(sum(row['bytes'] for row in raw.values())==2169655337 and sum(row['bytes'] for row in raw.values())<=3072*1024**2,
                'finite actual016 raw-byte envelope')
        for name, row in raw.items():
            path = root/name
            def hash_raw(path=path, row=row, name=name):
                require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
                        and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'raw NPZ SHA:' + name)
            audit.op('raw_NPZ_SHA', dict(raw_files_SHA_checked=1, raw_bytes_SHA_read=row['bytes']), hash_raw, path=name)
            audit.touched[name] = dict(bytes=row['bytes'], sha256=row['sha256'])
            audit.guard()
        stats_path = Path(cfg['inputs']['statistics']['path'])
        require(stats_path.suffix == '.json' and sha(stats_path) == cfg['inputs']['statistics']['sha256'], 'frozen small statisticsJSON')
        stats = load(stats_path)['loss']
        qfloor, peakfloor = float(stats['q_floor']), float(stats['peak_floor'])
        kernel_path = Path(__file__).resolve().with_name('saved_array_kernels.py')
        require(sha(kernel_path) == KERNEL_SHA, 'registered pure kernel SHA')
        global np, kernels
        import numpy as np
        spec = importlib.util.spec_from_file_location('saved_only_kernels014', kernel_path)
        kernels = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernels)
        require('torch' not in sys.modules and 'n01' not in sys.modules, 'no neural imports')
        for trial in TRIALS:
            for snapshot, arms in [('initial_evaluation', ('A', 'M0', 'initial')), ('final_evaluation', ('final',))]:
                baseline = audit.load_baselines(trial, snapshot)
                for bank in (0,1):
                    audit.op('charge_bank_saved_statistics',dict(charge_npz_files_decoded=1,charge_A_rows_verified=N,
                        charge_A_float64_cells_verified=N*384,charge_mu_draws_verified=N,charge_q_draws_verified=N,
                        charge_parent_moment_records_verified=P),lambda bank=bank:audit.charge_bank(trial,snapshot,bank),
                        trial=trial,snapshot=snapshot,bank=bank)
                for arm in arms:
                    for h in H:
                        audit.op('two_bank_pair_verified', dict(bank_pairs_verified=1),
                                 lambda: audit.pair(trial, snapshot, arm, h, baseline, qfloor, peakfloor),
                                 trial=trial, snapshot=snapshot, arm=arm, hypothesis=h)
                        if not (audit.out/'FIRST_PAIR_PROOF.json').exists():
                            write_new(audit.out/'FIRST_PAIR_PROOF.json', dict(audit.base(),
                                schema='n01/wp3-readback-first-formal-pair/v1',
                                actual_saved_array_pair_completed=True, pair=audit.pairs[0],
                                no_extra_tiny_actual_input=True, actual_new_CP_NN_AD=0))
                del baseline
                gc.collect()
        require(len(audit.pairs)==24 and len(audit.charge_banks)==8 and not audit.probes,'complete016 finite readback without probes')
        require(not audit.inflight and audit.counts==audit.attempted==EXPECTED_COUNTS,'exact016 complete counters')
        require(audit.entered==audit.completed==200,'exact016200 operations')
        result = dict(audit.base(), schema='n01/wp3-independent-raw-saved-array-readback/v1', status='PASS', passed=True,
            UTC=datetime.now(timezone.utc).isoformat(), frozen_configuration_sha256=CONFIG_SHA,
            frozen_manifest_sha256=MANIFEST_SHA, inventory_sha256=sha(args.inventory), raw_files_SHA_checked=audit.touched,
            pairs=audit.pairs, charge_banks=audit.charge_banks, probes=audit.probes, numerical_comparison_absmax_errors=audit.errors,
            decoded_window_hashes=audit.raw_hashes, actual_baseline_hashes=audit.baseline_hashes,
            resources=audit.guard(), original_Y_second_copy_available=False,
            native_complete_wave_QPSD_noise_distances_independently_replayed=False,
            distance_evidence='frozen producer + separately verified JSON aggregation/decision only; no raw CDF/loss replay',
            original_C1_median_fit_independently_replayed=False,
            preADC384_and_fullcarrier_independent_readback=False,
            frozen_A_preADC384_independently_recomputed_from_saved_rows=True,
            full_stochastic_preADC384_carrier_independently_replayed=False,
            original_SOURCE_input_byte_hash_independently_replayed=False,
            saved_input_digest_repetition_and_crossbank_identity_only=True,
            charge_latent_or_RNG_replay=False,charge_r8_added=False,
            probe_or_gradient_readback_applicable=False,
            charge_mean_protection_is_not_physical_response_identification=True,
            reviewer_operations=dict(CP_loads=0, NN_rows=0, AD_queries=0, training_updates=0,
                optimizer_or_sampler_created=0, new_random_draws=0, CDF_or_loss_replays=0,
                original_TRAIN_or_DEV_waveform_reads=0, GPU_work=0),
            scientific_acceptance=False, scientific_status='response_match_not_established')
        write_new(audit.out/'ARRAY_READBACK.json', result)
        audit.phase = 'complete'
        write_new(audit.out/'COMPLETE.json', dict(audit.base(), native_status='COMPLETE', passed=True,
            ARRAY_READBACK_sha256=sha(audit.out/'ARRAY_READBACK.json'), scientific_acceptance=False))
        audit.state(dict(native_status='COMPLETE', passed=True))
        print(json.dumps(dict(status='PASS', receipt=str(audit.out/'ARRAY_READBACK.json'),
                              sha256=sha(audit.out/'ARRAY_READBACK.json'), resources=audit.guard())), flush=True)
    except BaseException as error:
        failure = dict(audit.base(), status='FAILED', native_status='FAILED', passed=False, error_type=type(error).__name__,
            error=str(error), pairs_completed=audit.pairs, probes_completed=audit.probes,
            raw_files_SHA_checked=audit.touched, no_retry_or_expansion=True,
            reviewer_scientific_CP_NN_AD_training=0)
        write_new(audit.out/'FAILED.json', failure)
        audit.state(dict(native_status='FAILED', passed=False, error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
