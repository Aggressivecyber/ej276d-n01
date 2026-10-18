from pathlib import Path
import ast
r=Path('E:/GAN/research/N01')
s=(r/'scripts/run_response069.py').read_text().replace('069','070').replace('level_warp070','split_warp070')
s=s.replace('from run_response068 import voltage_warp068','from run_response069 import level_warp069')
s=s.replace('One conditional crossing-level voltage warp with area-preserving random coupling.','One crossing-level warp with separate front/tail integral constraints.')
s=s.replace('Conditional50/90% crossing queries, one area scale, linear residual projection.','069 crossing queries; two positive-area scales and two linear residual projections.')
a=s.index('    scale=area/interpolated.sum(1);mapped=scale[:,None]*interpolated');b=s.index('    rows=dict(',a)
s=s[:a]+'''    front=(positions+110)<peak[:,None]+30
    areas=torch.stack(((positive*front).sum(1),(positive*~front).sum(1)),1)
    mapped_areas=torch.stack(((interpolated*front).sum(1),(interpolated*~front).sum(1)),1)
    if not bool((areas>0.).all() and (mapped_areas>0.).all()):raise ValueError('070 empty positive front/tail area domain')
    scales=areas/mapped_areas;scale=torch.where(front,scales[:,0,None],scales[:,1,None]);mapped=scale*interpolated
    delta=torch.zeros_like(mean);delta[:,110:176]=mapped-positive;adapted=mean+delta
    active=torch.where(mask,random[:,110:176],torch.zeros_like(local))
    provisional=scale*torch.where(mask,(1.-weight)*active.gather(1,lo)+weight*active.gather(1,lo+1),torch.zeros_like(local))
    leakage_parts=torch.stack((((provisional-active)*front).sum(1),((provisional-active)*~front).sum(1)),1)
    leakage=torch.where(front,leakage_parts[:,0,None],leakage_parts[:,1,None]);unit=mapped/torch.where(front,areas[:,0,None],areas[:,1,None])
    random_delta=torch.zeros_like(random);random_delta[:,110:176]=provisional-leakage*unit-active
    split=peak+30
'''+s[b:]
s=s.replace('amplitude_scale=scale,','front_amplitude_scale=scales[:,0],tail_amplitude_scale=scales[:,1],')
s=s.replace('random_provisional_integral_leakage_Vns=leakage)', '''random_provisional_integral_leakage_Vns=leakage_parts.sum(1),
        reference_split_index=split,positive_front_area_Vns=areas[:,0],positive_tail_area_Vns=areas[:,1],interpolated_positive_front_area_Vns=mapped_areas[:,0],interpolated_positive_tail_area_Vns=mapped_areas[:,1],
        reference_front_integral_residual_Vns=(delta[:,110:176]*front).sum(1),reference_tail_integral_residual_Vns=(delta[:,110:176]*~front).sum(1),
        random_reference_front_integral_residual_Vns=(random_delta[:,110:176]*front).sum(1),random_reference_tail_integral_residual_Vns=(random_delta[:,110:176]*~front).sum(1),
        random_front_provisional_leakage_Vns=leakage_parts[:,0],random_tail_provisional_leakage_Vns=leakage_parts[:,1],
        original_split_step_V=(mean.gather(1,split[:,None])-mean.gather(1,(split-1)[:,None])).squeeze(1),
        new_split_step_V=(adapted.gather(1,split[:,None])-adapted.gather(1,(split-1)[:,None])).squeeze(1),
        correction_split_step_V=(delta.gather(1,split[:,None])-delta.gather(1,(split-1)[:,None])).squeeze(1))''')
s=s.replace("progress['level_warp_records']+=1;progress['level_warp_rows']+=len(q)","progress['split_warp_records']+=1;progress['split_warp_rows']+=len(q)")
s=s.replace('correction,random_delta,rows,_=voltage_warp068(fixed_mean,random,q,template)','correction,random_delta,rows,_,_,_=level_warp069(fixed_mean,random,q,template)')
s=s.replace("progress['voltage_warp_records']+=1;progress['voltage_warp_rows']+=len(q)","progress['level_warp_records']+=1;progress['level_warp_rows']+=len(q)")
s=s.replace("if float(rows['query_minimum_spacing_samples'].min())<0.","if max(float(rows[k].abs().max()) for k in ('reference_front_integral_residual_Vns','reference_tail_integral_residual_Vns','random_reference_front_integral_residual_Vns','random_reference_tail_integral_residual_Vns'))>1e-12:raise RuntimeError('070 split-integral conservation failed')\n        if float(rows['query_minimum_spacing_samples'].min())<0.")
s=s.replace('crossing_level_warp_enabled=enabled','crossing_level_warp_enabled=True,split_area_warp_enabled=enabled')
s=s.replace('WARP068','LEVEL069').replace('LEVELWARP','SPLITWARP')
s=s.replace("expected['VOLTAGEWARP']['metrics']","expected['LEVELWARP']['metrics']")
s=s.replace("replay[str(index)]['VOLTAGEWARP']['metrics']","replay[str(index)]['LEVELWARP']['metrics']")
s=s.replace("('SPLITWARP',p['level_warp_enabled'])","('SPLITWARP',p['split_warp_enabled'])")
s=s.replace("crossing_level_warp_enabled=p['level_warp_enabled'],","crossing_level_warp_enabled=True,split_area_warp_enabled=p['split_warp_enabled'],reference_split_peak_offset=30,")
s=s.replace('response070-crossing-level-voltage-warp-result','response070-split-area-crossing-level-warp-result')
s=s.replace('CLOSED_NEGATIVE_CROSSING_LEVEL_VOLTAGE_WARP','CLOSED_NEGATIVE_SPLIT_AREA_CROSSING_LEVEL_WARP')
s=s.replace('fixed_crossing_level_voltage_warp','fixed_split_area_crossing_level_warp').replace('first_head_crossing_level_warp_complete','first_head_split_area_warp_complete')
s=s.replace('crossing_level_warp_enabled=True,support=', 'crossing_level_warp_enabled=True,split_area_warp_enabled=True,reference_split_peak_offset=30,support=')
s=s.replace('068 complete baseline','069 complete baseline').replace('068 native replay','069 native replay').replace('068 replay','069 replay')
s=s.replace('one positive-area scale.', 'separate positive-area scales before/from reference mean peak+30. The split may introduce a seam; reference integrals are not native PSD guarantees.')
s=s.replace('deterministic linear integral projection.', 'deterministic linear front/tail integral projections.')
ast.parse(s);(r/'scripts/run_response070.py').write_text(s)
t=(r/'scripts/test_response069.py').read_text().replace('069','070').replace('level_warp070','split_warp070')
t=t.replace('Independent manufactured crossing-level warp and residual-integral checks.','Independent manufactured crossing queries, split mean and residual-integral checks.')
t=t.replace('m[:,:100]=0.;','m[:,:100]=0.;m[:,151:176]+=.001;')
a=t.index(' positive=old[110:176]*mask;');b=t.index(' assert np.array_equal(d[i,110:176]',a)
t=t[:a]+''' positive=old[110:176]*mask;v=np.interp(expected_query,np.arange(110,176),positive)*mask;active=rnd[110:176]*mask;raw_random=np.interp(expected_query,np.arange(110,176),active)*mask
 mapped=np.zeros(66);rr=np.zeros(66);front=np.arange(110,176)<peak+30
 for side,select in [('front',front),('tail',~front)]:
  alpha=positive[select].sum()/v[select].sum();mapped[select]=alpha*v[select];part=alpha*raw_random[select]
  rr[select]=part-(part.sum()-active[select].sum())*mapped[select]/positive[select].sum()-active[select]
  scale_err=max(scale_err,abs(alpha-float(rows[side+'_amplitude_scale'][i])))
 mean_err=max(mean_err,float(np.max(np.abs(mapped-positive-d[i,110:176].cpu().numpy()))));random_err=max(random_err,float(np.max(np.abs(rr-dr[i,110:176].cpu().numpy()))))
 query_err=max(query_err,float(np.max(np.abs(expected_query-query[i].cpu().numpy()))))
 assert int(rows['reference_split_index'][i])==peak+30
 for select in (front,~front):assert abs(d[i,110:176].cpu().numpy()[select].sum())<1e-12 and abs(dr[i,110:176].cpu().numpy()[select].sum())<1e-12
 assert abs(float(rows['correction_split_step_V'][i])-(d[i,peak+30]-d[i,peak+29]).item())<1e-15
'''+t[b:]
t=t.replace('dd,_,ds,dq,_,_=split_warp070','dual[0,150:175]=.001\ndd,_,ds,dq,_,_=split_warp070')
t=t.replace('td,_,_,tq,_,_=split_warp070','tri[0,150:175]=.001\ntd,_,_,tq,_,_=split_warp070')
t=t.replace("assert caught\nr.write_json", "assert caught\nempty=torch.zeros_like(tri);empty[0,126]=1.;caught_empty=False\ntry:split_warp070(empty,torch.zeros_like(empty),torch.tensor([1.],device=device,dtype=torch.float64),triangle)\nexcept ValueError as exc:assert 'empty positive front/tail area' in str(exc);caught_empty=True\nassert caught_empty\nr.write_json")
t=t.replace('mean_full_integral_error_Vns=', "mean_split_integral_max_error_Vns=max(float(rows[k].abs().max()) for k in ('reference_front_integral_residual_Vns','reference_tail_integral_residual_Vns')),random_split_integral_max_error_Vns=max(float(rows[k].abs().max()) for k in ('random_reference_front_integral_residual_Vns','random_reference_tail_integral_residual_Vns')),empty_partition_fails_explicitly=True,\n mean_full_integral_error_Vns=")
ast.parse(t);(r/'scripts/test_response070.py').write_text(t)
print('070 split-area implementation and manufactured tests prepared')
