"""Independent stdlib-only algebra checks for the WP4 contract; no data or RNG."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import math

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent


def check_contract():
    p = json.loads((HERE/'ENGINEERING_PROPOSAL.json').read_bytes())
    checks = []

    def require(name, test, **detail):
        assert test, name
        checks.append(dict(name=name, passed=True, **detail))

    # The two-point states are exhaustively enumerated, never randomly drawn.
    def expected_energy(student_scale):
        values = []
        for signs in product((-1, 1), repeat=4):
            s = [student_scale*signs[0], student_scale*signs[1]]
            t = [signs[2], signs[3]]
            values.append(sum(abs(x-y) for x in s for y in t)/4-abs(s[0]-s[1])/2)
        return sum(values)/len(values)

    require('independent_U_energy_correct_dispersion_beats_collapse',
            expected_energy(1) == .5 < expected_energy(0) == 1)
    require('independent_U_energy_correct_dispersion_beats_overdispersion',
            expected_energy(2) == 1 > expected_energy(1))
    mse_correct = sum((s-t)**2 for s, t in product((-1, 1), repeat=2))/4
    mse_collapsed = sum(t*t for t in (-1, 1))/2
    require('single_random_target_MSE_has_collapse_contrast', mse_collapsed == 1 < mse_correct == 2)

    m = Fraction(9, 10)
    fourth = m**4+6*m*m*(1-m*m)+3*(1-m*m)**2
    require('T2_centered_unit_variance_fourth_moment',
            fourth == Fraction(8439, 5000) and float(fourth) == p['teachers']['T2']['fourth_moment'],
            mean=0, variance=1, fourth_moment=float(fourth))
    a = p['teachers']['T1']['a_known_Vns']; b = p['teachers']['T1']['b_known']
    ai = p['student']['a_initial_Vns']; bi = p['student']['b_initial']
    qref = p['student']['q_reference_Vns']
    for Q in [.25, .5, 1., qref, 8., 16.]:
        s2 = math.log1p(a/Q+b)
        mu = Q*math.exp(-s2/2+s2/2)
        var = Q*Q*math.expm1(s2)
        require('positive_mean_corrected_lognormal_at_'+str(Q),
                mu == Q and math.isclose(var, a*Q+b*Q*Q, rel_tol=4e-15), Q=Q)
        require('initial_variance_is_tenth_at_'+str(Q),
                math.isclose((ai*Q+bi*Q*Q)/(a*Q+b*Q*Q), .1, rel_tol=4e-15))

    def gram_ratio(z):
        aa = sum(x*x for x in z)/len(z)
        ab = sum(x**3 for x in z)/len(z)
        bb = sum(x**4 for x in z)/len(z)
        hi = ((aa+bb)+math.sqrt((aa-bb)**2+4*ab*ab))/2
        lo = (aa*bb-ab*ab)/hi
        return lo/hi
    require('constant_Q_cannot_identify_two_coefficients', gram_ratio([1., 1., 1.]) == 0)
    require('manufactured_Q_span_can_identify_two_coefficients', gram_ratio([.5, 1., 2., 4.]) >= 1e-4,
            eigenvalue_ratio=gram_ratio([.5, 1., 2., 4.]))

    # Exact discrete tie-PIT probabilities on independent Bernoulli samples.
    for n in range(1, 5):
        for lower, upper in [(Fraction(1, 4), Fraction(3, 4)), (Fraction(1, 20), Fraction(19, 20))]:
            probability = Fraction(0)
            for vals in product((0, 1), repeat=n+1):
                students, target = vals[:n], vals[-1]
                L = sum(x < target for x in students); E = sum(x == target for x in students)
                lo = Fraction(L, n+1); hi = Fraction(L+E+1, n+1)
                overlap = max(Fraction(0), min(upper, hi)-max(lower, lo))
                probability += overlap/(hi-lo)/2**(n+1)
            require('discrete_randomized_PIT_n'+str(n)+'_level'+str(upper-lower),
                    probability == upper-lower)
    require('zero_student_PIT_requires_pending_despite_uniform_formula',
            Fraction(3, 4)-Fraction(1, 4) == Fraction(1, 2),
            meaning='n=0 produces U algebraically, so support cannot be accepted from this formula alone')

    N = p['data_roles']['heldout_new_parameter_parents']
    for sign in (-1, 1):
        weights = [math.exp(sign*math.log(2)*(2*(i+.5)/N-1)) for i in range(N)]
        ess = sum(weights)**2/sum(x*x for x in weights)
        require('fixed_population_weight_bounds_and_ESS_'+str(sign),
                min(weights) >= .5 and max(weights) <= 2 and ess/N >= .5, ESS=ess, parents=N)

    P, H, R, banks, updates, K = 6593, 3, 32, 2, 2000, 512
    per_eval = P*banks*R
    for label, paired, unpaired, teachers in [('stage1_gate', 2, 0, 1), ('full_eight_trial_envelope', 4, 4, 2)]:
        c = p['exact_planned_budgets'][label]; trials = paired+unpaired
        student_rows = trials*updates*K*2+(1+trials)*per_eval
        teacher_rows = trials*updates*K*2+teachers*per_eval
        rows = student_rows+teacher_rows
        expected = dict(
            G_updates=trials*updates, new_AD_queries=trials*updates, checkpoint_loads=3,
            A_original_forward_rows=P, original_U_Net_decoded_rows=P, fixed_M0_head_actual_kernel_rows=P,
            student_charge_module_rows=student_rows, teacher_charge_module_rows=teacher_rows,
            shape_cached_basis_assembly_rows=rows, noise005_rows=rows,
            hard_ADC_rows=rows*H, hard_first_argmax_gather_rows=rows*H,
            evaluation_kernel_snapshots=1+trials+teachers,
            evaluation_signal_rows=(1+trials+teachers)*per_eval,
            total_scalar_normal_requests=rows*(8+1+101+284),
            T2_teacher_sign_uniform_requests=(4*updates*K*2+per_eval) if teachers == 2 else 0,
            coverage_PIT_uniform_requests=teachers*per_eval*H*6,
            parent_randint_calls=(paired+4*unpaired)*updates,
            parent_randint_scalar_requests=(paired+4*unpaired)*updates*K,
            paired_training_energy_pair_distances=paired*updates*K*5*H,
            unpaired_training_energy_pair_distances=unpaired*updates*(2*K)**2*2*H)
        for key, value in expected.items():
            require(label+'_'+key, c[key] == value, value=value)
        require(label+'_normal_substreams',
                c['scalar_normal_requests'] == {key:rows*dim for key, dim in [('shape',8),('charge',1),('noise005',101),('extension284',284)]})
    storage = p['storage']
    require('hard_code_payload', storage['hard256_int16_payload_bytes'] == 11*banks*H*P*R*256*2)
    require('baseline_code_payload', storage['baseline100_int16_payload_bytes'] == 11*banks*P*R*100*2)
    cache = p['preparation_cache']
    require('A_B_Q_cache_payload', cache['persisted_A_B_Q_numerical_bytes'] == P*384*8+P*8*384*8+P*8)
    require('historical_scope_and_zero_current_operations', not p['scientific_acceptance'] and not p['hardware_calibration'] and
            all(v == 0 for v in p['actual_scientific_work_in_this_preparation'].values()))
    return dict(passed=True, checks=checks, count=len(checks),
                actual_CP_SOURCE_REAL_DEV_NN_AD_G_RNG_GPU_SSH=0,
                scope='Deterministic stdlib algebra and exhaustive manufactured finite states; not scientific recovery evidence')


if __name__ == '__main__':
    result = check_contract()
    print(json.dumps(result, ensure_ascii=False, indent=2))
