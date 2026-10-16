from pathlib import Path
import time,json
r=Path('/home/rylan/projects/ej276d-n01/wp3_024_20261016');a=r/'runs/primary/artifacts';sv=r/'runs/primary/supervisor';deadline=time.monotonic()+45
while not ((a/'FIRST_ACTUAL_HARD_PROOF.json').exists() or (a/'FAILED.json').exists() or (sv/'exit_receipt.json').exists()):
 if time.monotonic()>=deadline:break
 time.sleep(.5)
print(json.dumps(dict(first_actual_base=(a/'FIRST_ACTUAL_BASE_PROOF.json').exists(),first_actual_hard=(a/'FIRST_ACTUAL_HARD_PROOF.json').exists(),driver_failed=(a/'FAILED.json').exists(),native_exit=(sv/'exit_receipt.json').exists())))
