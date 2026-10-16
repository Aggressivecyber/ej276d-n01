"""One two-parent/four-draw semantic check of the actual026 weighted-measure path."""
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root/'scripts'))
import numpy as np
from run_wp5_026 import finite_measure026

started = time.monotonic()
x = np.array([[0., 2., 4., 6.], [8., 10., 12., 14.]])
w = np.array([1., 3.])
means, within = x.mean(1), x.var(1, ddof=0)
full = finite_measure026(x, w, draws=4, native4=(means, within))
np.testing.assert_array_equal(means, [3., 11.])
np.testing.assert_array_equal(within, [5., 5.])
np.testing.assert_allclose([full['mean'], full['variance'], full['effective_weight_units_ESS'],
    full['finite_native4_decomposition']['between'], full['finite_native4_decomposition']['within'],
    full['finite_native4_decomposition']['total']], [9., 17., 1.6, 12., 5., 17.], rtol=1e-14, atol=0.)
masked = x.copy()
masked[0, [1, 3]] = np.nan
masked[1, 2] = np.nan
partial = finite_measure026(masked, w, draws=4)
np.testing.assert_allclose([partial['mean'], partial['variance'], partial['defined_raw_comparison_weight_mass'],
    partial['defined_comparison_weight_mass_fraction'], partial['effective_weight_units_ESS']],
    [100/11, 2056/121, 11/4, 11/16, 121/85], rtol=1e-14, atol=0.)
assert partial['defined_rows'] == 5 and partial['undefined_rows'] == 3 and partial['defined_weight_units'] == 2
empty = finite_measure026(np.full((2, 4), np.nan), w, draws=4)
assert empty['mean'] is empty['variance'] is empty['effective_weight_units_ESS'] is None
assert empty['defined_rows'] == 0 and empty['defined_comparison_weight_mass_fraction'] == 0
source = (root/'scripts/run_wp5_026.py').read_bytes()
numeric = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'finite_measure026')
memory = ctypes.create_string_buffer(72)
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(memory), len(memory)):
    raise ctypes.WinError()
out = root/'receipts/wp5_026_20261016/engineering_01/fixture_01'
out.mkdir(exist_ok=False)
receipt = dict(schema='n01/wp5-026-artificial-finite-measure-check/v1', status='PASS',
    driver_sha256=hashlib.sha256(source).hexdigest(),
    numeric_function_AST_sha256=hashlib.sha256(ast.dump(numeric, include_attributes=False).encode()).hexdigest(),
    fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    artificial_only=True, parents=2, draws=4, finite_measure_evaluations=3, artificial_scalar_row_evaluations=24,
    unequal_raw_parent_weights=[1, 3], analytic_full_mean=9., analytic_within=5., analytic_between=12., analytic_total=17.,
    finite_mask_defined_rows=5, finite_mask_parent_ESS='121/85', full=full, partial=partial, empty=empty,
    native4_within_denominator=4, scientific_private_array_reads=0, NN_CP_AD_G_RNG_CDF_increment=0,
    wall_seconds=time.monotonic()-started, peak_working_set_bytes=ctypes.c_size_t.from_buffer(memory, 8).value,
    memory_measurement_stage='before_fixture_receipt_serialization')
(out/'RESULT.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf8')
print(json.dumps(dict(status='PASS', fixture=str(out/'RESULT.json'), artificial_scalar_rows=24)))
