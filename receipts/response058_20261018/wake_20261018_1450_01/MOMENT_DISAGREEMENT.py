from pathlib import Path
import json,math
import numpy as np
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;o=w/'collected/run_01'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=[]
for seed in (0,1):
    for case in ('BASE054','PROBE_1_neg','PROBE_1_pos','ENDPOINT'):
        report=read(o/f'seed{seed}_{case}.json')
        for h,m in report['metrics'].items():
            bins=m['output_Q_conditions']['bins'];moment={}
            for population in ('fake','real'):
                n=np.array([z[population]['FWHM_ns']['defined_rows'] for z in bins],np.int64)
                mean=np.array([z[population]['FWHM_ns']['mean'] for z in bins]);var=np.array([z[population]['FWHM_ns']['std_population']**2 for z in bins])
                weights=n/n.sum();pooled_mean=float(weights@mean);within=float(weights@var);between=float(weights@((mean-pooled_mean)**2))
                moment[population]=dict(defined_rows=int(n.sum()),undefined_rows=sum(z[population]['FWHM_ns']['undefined_rows'] for z in bins),mean_ns=pooled_mean,
                    within_own_Q_bin_variance_ns2=within,between_own_Q_bin_variance_ns2=between,total_variance_ns2=within+between,
                    bin_defined_counts=n.tolist(),bin_means_ns=mean.tolist(),bin_variances_ns2=var.tolist())
                if population=='fake' and case in ('BASE054','ENDPOINT'):
                    z=np.load(o/f'seed{seed}_{case}_{h}_FEATURE_ROWS.npz')['FWHM_ns'];z=z[np.isfinite(z)]
                    assert len(z)==n.sum() and abs(z.mean()-pooled_mean)<1e-12 and abs(z.var()-(within+between))<1e-12
            real_weights=np.array(moment['real']['bin_defined_counts'])/moment['real']['defined_rows']
            conditional_mean_gap=float(real_weights@np.abs(np.array(moment['fake']['bin_means_ns'])-np.array(moment['real']['bin_means_ns'])))
            rows.append(dict(seed=seed,hypothesis=h,case=case,original_marginal_FWHM_W1_ns=m['values']['FWHM_ns'],
                REAL_defined_count_weighted_conditional_mean_absolute_gap_ns=conditional_mean_gap,moments=moment,
                interpretation='Own observed-Q grouping; descriptive moment identity only, not paired energy truth or a replacement acceptance metric.'))
out=dict(rows=rows,full_BASE_and_ENDPOINT_population_variance_identity_verified=True,
    probe_scope='Probe moment decomposition derived exactly from stored bin counts, means and population SD; individual probe CDFs/crossings were not saved in058.',
    no_new_model_calls=True,no_new_candidate=True,no_contract_change=True)
(w/'MOMENT_DISAGREEMENT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([dict(case=z['case'],marginal_W1_ns=z['original_marginal_FWHM_W1_ns'],conditional_gap_ns=z['REAL_defined_count_weighted_conditional_mean_absolute_gap_ns'],
    SOURCE=z['moments']['fake'],REAL=z['moments']['real']) for z in rows if z['seed']==0 and z['hypothesis']=='fast'],indent=2))
