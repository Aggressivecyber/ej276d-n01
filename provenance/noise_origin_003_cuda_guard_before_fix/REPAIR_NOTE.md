# CUDA device guard compatibility repair (2026-10-12)

The first real_real attempt exited natively with code1 before reading target or fitting a classifier: torch.cuda.set_per_process_memory_fraction rejected torch.device('cuda') because its device index was unspecified. The origin phase had already completed21047 rows with bitwise target equality and must not be repeated. REAL-REAL scientific classifier updates before this error:0; noise/response updates:0.

Both new RR calibration and post-run replay scripts now resolve the existing --device cuda option to torch.device('cuda:0'). CUDA_VISIBLE_DEVICES=0 still selects the same exposed GPU. No sampling, seed, batch, architecture, optimizer, bootstrap, permutation, budget or statistical method changed. The continuation restarts RR at index0 for all128 fixed fits; the failed parent run and original source binding remain preserved. No real run was launched while applying this compatibility repair.

Preserved original SHA256:

- calibrate_real_real_null.py:11930d3b9680c22506f06d20148918a16e14e4d2a3e541186b42dfcb62612f6a
- verify_noise_origin_results.py:44d58841a6c07221643690a7e68708a9215cfe88b361843769310339ce49c52c
- REAL_REAL_NULL_METHOD.md:7a880f513e224d333f7a265df80bea88df1c89caabaf3b59f367ef551288f1e3

All three originals are stored alongside this note. New source hashes must be bound to the separate continuation_001 run; do not rewrite the failed parent execution receipt.
