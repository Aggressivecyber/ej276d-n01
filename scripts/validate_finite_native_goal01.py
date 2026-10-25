"""P0: exact finite model, actual original constraints and complete CDF cost."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np
import torch
from finite_native_goal01 import FiniteNativeGoal01

parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--config-sha256',required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();started=time.monotonic()
assert hashlib.sha256(args.config.read_bytes()).hexdigest()==args.config_sha256
p=json.loads(args.config.read_bytes());args.out.mkdir(exist_ok=False)
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)

def progress(stage,**extra):
    assert time.monotonic()-started<p['resources']['internal_seconds']
    (args.out/'PROGRESS.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,elapsed_seconds=time.monotonic()-started,**extra))+'\n')

try:
    identity=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),argv=sys.argv,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[19],config_sha256=args.config_sha256,parameter_fits=0,candidate_selections=0,automatic_restart=False)
    (args.out/'START.json').write_text(json.dumps(identity,indent=2)+'\n')
    for name,rec in {**p['inputs'],**p['frozen_source_pins'],**p['new_source_pins']}.items():
        path=Path(name)
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        assert path.stat().st_size==rec['bytes'] and digest==rec['sha256'],name
    progress('all_inputs_and_source_pins_verified')
    model=FiniteNativeGoal01(p,progress)
    records=[]
    for label,point,readout_path in [('M0',np.zeros(163),p['original_readout']),('fixed_zero_warmstart',model.warmstart,p['known_point_readout'])]:
        progress('full_point_start',case=label)
        record=model.evaluate(point,keep_arrays=True)
        arrays=record.pop('readout');means=record.pop('memory_means')
        with np.load(readout_path,allow_pickle=False) as z:
            if label=='M0':reference={key:z[key][model.cache['generated_indices']] for key in arrays}
            else:
                np.testing.assert_array_equal(z['global_generated_indices'],model.cache['generated_indices'])
                reference={key:z[key] for key in arrays}
        for key in arrays:np.testing.assert_array_equal(arrays[key],reference[key])
        expected=model.source['native']['original' if label=='M0' else 'candidate']['values']
        errors={key:abs(record['values'][key]-expected[key]) for key in ['wave','QPSD']}
        assert max(errors.values())<1e-9
        risk_error=max(abs(record['metrics_by_bank'][bank][name]-model.source['records']['original' if label=='M0' else 'candidate']['metrics_by_bank'][bank][name]) for bank in range(2) for name in p['risk_names'])
        assert risk_error<1e-9
        for bank in range(2):assert record['flags'][bank]==model.source['records']['original' if label=='M0' else 'candidate']['flags'][bank]
        if label=='M0':np.testing.assert_allclose(means,model.saved['original'],atol=1e-12,rtol=0)
        record.update(case=label,ADC_window_prefix_crop_and_flags_exact=True,native_value_errors=errors,max_risk_error=risk_error)
        records.append(record)
        (args.out/(label+'_REPLAY.json')).write_text(json.dumps(record,indent=2)+'\n')
        del arrays,reference,means
        progress('full_point_passed',case=label,point_seconds=record['seconds'])
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS_EXACT_FINITE_MODEL_ACTUAL_GUARDS_AND_COMPLETE_NATIVE_CDF',job_number=p['job_number'],cases=records,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(0),cached_observation_rows=206880,complete_wave_columns=17552,complete_QPSD_columns=52,SOURCE_FIT_parents=6465,REAL_FIT_rows=18948,parameter_fits=0,candidate_selections=0,optimizer_updates=0,new_stochastic_draws=0,original_backbone_calls=0,REPORT_rows_used_in_metrics=0,NAS_access=False,scientific_acceptance=False,restricted_average_response_acceptance=False)
    assert result['peak_RSS_bytes']<12*1024**3 and result['peak_GPU_reserved_bytes']<=4*1024**3
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE'))+'\n')
    print(json.dumps(result),flush=True)
finally:
    error=sys.exc_info()[1]
    if error is not None:(args.out/'FAILED.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started,parameter_fits=0,candidate_selections=0,automatic_restart=False))+'\n')
