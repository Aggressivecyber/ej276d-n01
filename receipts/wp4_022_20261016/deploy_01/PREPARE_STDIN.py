from pathlib import Path; import sys; r=Path(sys.argv[1]); s=Path(sys.argv[2]); assert not r.exists(), 'preserve existing022'; s.mkdir(exist_ok=False); print(s)
