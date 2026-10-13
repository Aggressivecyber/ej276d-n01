"""Package an artificial capacity test without any scientific data/weights."""
from pathlib import Path
import hashlib
import json
import tarfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/response_007_20261013/engineering01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    paths=sorted(list((ROOT/'src').rglob('*.py'))+[ROOT/'tests/test_response007.py',ROOT/'response007_config.json',
        ROOT/'scripts/benchmark_response007.py',ROOT/'scripts/supervise.py'])
    files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    package=OUT/'BENCHMARK_SOURCE.tar.gz'
    with tarfile.open(package,'w:gz') as t:
        for p in paths:t.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    receipt=dict(scope='fabricated_waveforms_only_no_scientific_weights_or_datasets',files=files,
        package_sha256=sha(package),package_bytes=package.stat().st_size)
    (OUT/'SOURCE_BINDING.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(package=str(package),sha256=sha(package),bytes=package.stat().st_size)))
if __name__=='__main__':main()
