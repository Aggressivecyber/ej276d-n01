"""Repeated finite163-coordinate model and actual hard-objective evaluation."""
from collections import Counter
from pathlib import Path
import json
import time

import numpy as np
import torch

from n01.global_memory01 import GlobalDistributionMemory01
from n01.losses008 import _wave_views
from n01.losses008_v2 import VIEWS, COORDINATE_VIEWS, SHORT_LAGS, ProjectionContext, empirical_wasserstein_1d
from n01.observation008 import observe_existing_carrier, observe_generated_carrier
from n01.response008 import loss_tensor_statistics
from run_response036 import NativeMetricLoss036


@torch.no_grad()
def complete_wave_value(fake, real, loss, context, progress):
    """All8776 original columns, bounded storage, no linearized target."""
    parts = dict(swd=0., coordinate=0., short_lag=0.)
    columns_done = 0
    for name in VIEWS:
        f = loss._z(name, _wave_views(fake, loss.peak_floor)[name])
        r = loss._z(name, _wave_views(real, loss.peak_floor)[name])
        groups = [('swd', None, context.banks['wave:'+name].to(f), 32, 6)]
        if name in COORDINATE_VIEWS:
            groups += [('coordinate', None, None, f.shape[1], 2)]
            groups += [('short_lag', lag, context.banks[f'lag:{name}:{lag}'].to(f), (f.shape[1]-lag)*4, 8) for lag in SHORT_LAGS]
        for kind, lag, directions, width, count in groups:
            value = 0.
            for first in range(0, width, 16):
                last = min(first+16, width)
                if kind == 'swd':
                    x, y = f@directions[:,first:last], r@directions[:,first:last]
                elif kind == 'coordinate':
                    x, y = f[:,first:last], r[:,first:last]
                else:
                    columns = torch.arange(first,last,device=f.device)
                    position, projection = columns//4, columns%4
                    a, b = directions[0,projection], directions[1,projection]
                    x = f[:,position]*a+f[:,position+lag]*b
                    y = r[:,position]*a+r[:,position+lag]*b
                value += float(empirical_wasserstein_1d(x,y))*(last-first)/width
            parts[kind] += value/count
            columns_done += width
            progress('complete_wave_group',view=name,kind=kind,lag=lag,completed_columns=columns_done,total_columns=8776)
        del f,r
    assert columns_done == 8776
    return .5*parts['swd']+.25*parts['coordinate']+.25*parts['short_lag'], parts


class FiniteNativeGoal01:
    """One frozen learned response; coefficients are the only fitted values."""
    def __init__(self, p, progress):
        self.p, self.progress = p, progress
        self.device = torch.device('cuda:0')
        self.source = json.loads(Path(p['source_native_result']).read_bytes())
        with np.load(p['mean_tangent'],allow_pickle=False) as z:
            self.mean = {key:z[key].copy() for key in ['decoded','raw_mean','logits','qbar','gate','source_FIT_rows','REAL_FIT_rows','original_mean','coordinate_steps','initial_head']}
        with np.load(p['additive_head'],allow_pickle=False) as z:
            self.fields = z['direction_fields'].copy()
            self.radii = z['mean_radii'].copy()
            self.additive_physical_steps = z['coordinate_steps'].copy()
            np.testing.assert_array_equal(z['source_FIT_rows'],self.mean['source_FIT_rows'])
            np.testing.assert_array_equal(z['REAL_FIT_rows'],self.mean['REAL_FIT_rows'])
        with np.load(p['source_joint_solution'],allow_pickle=False) as z:
            self.warmstart = z['primal'].copy()
            self.steps = z['coordinate_steps'].copy()
            np.testing.assert_array_equal(self.steps[:161],self.mean['coordinate_steps'])
            np.testing.assert_array_equal(self.steps[161:],np.ones(2))
        with np.load(p['finite_cache'],allow_pickle=False) as z:
            self.cache = {key:z[key].copy() for key in ['pre_ADC_carrier0','rows','coordinate_values','peak_reference','generated_indices','q_floor','original_theta']}
        assert len(self.warmstart)==163 and self.cache['pre_ADC_carrier0'].shape==(103440,384)
        lookup = np.full(6593,-1,dtype=np.int64)
        lookup[self.mean['source_FIT_rows']] = np.arange(6465)
        self.parent = lookup[self.cache['rows']]
        assert np.all(self.parent>=0) and np.all(np.bincount(self.parent,minlength=6465)==16)
        np.testing.assert_array_equal(self.mean['source_FIT_rows'][self.parent],self.cache['rows'])
        self.segments = []
        self.left = np.empty((103440,384),dtype=np.int16)
        self.fraction = np.empty((103440,384))
        self.multiplier = np.empty((103440,384))
        axis = np.arange(384,dtype=np.float64)
        indices = self.cache['generated_indices']
        for bank in range(2):
            for chunk in range(52):
                positions = np.flatnonzero((indices//52744==bank)&((indices%52744)//1024==chunk))
                for first in range(0,len(positions),256):
                    take = positions[first:first+256]
                    self.segments.append((bank,take))
                    a = self.cache['coordinate_values'][take]
                    peak = self.cache['peak_reference'][take,None]
                    mid = axis[101:][None]-.5
                    speed = np.exp(a[:,1:2]*np.exp(-.5*((mid-peak+8)/8)**2)+a[:,2:3]*np.exp(-.5*((mid-peak-18)/22)**2)+a[:,4:5]/(1+np.exp(-(mid-peak-55)/15)))
                    integral = np.cumsum(speed,axis=1)
                    queries = np.c_[np.broadcast_to(axis[:101],(len(take),101)),100+283*integral/integral[:,-1:]]
                    left = np.minimum(np.floor(queries).astype(np.int16),382)
                    self.left[take] = left
                    self.fraction[take] = queries-left
                    self.multiplier[take] = np.exp(a[:,0:1]+a[:,3:4]/(1+np.exp(-(axis[None]-peak-30)/8))+a[:,5:6]*np.exp(-.5*((axis[None]-peak)/4)**2))
        definition = json.loads(Path(p['metric_definition']).read_bytes())
        with np.load(p['metric_state'],allow_pickle=False) as z:
            state = {key:torch.from_numpy(z[key].copy()) for key in z.files}
        self.loss = NativeMetricLoss036(loss_tensor_statistics(definition['statistics']),definition['integral_statistics'],seed=2026101429,evaluation_seed=definition['projection_context']['seed'],integral_seed=2026101431)
        self.loss.load_state_dict({key[5:]:value for key,value in state.items() if key.startswith('loss:')},strict=True)
        self.loss = self.loss.to(self.device)
        self.context = ProjectionContext('evaluation',definition['projection_context']['seed'],None,{key[11:]:value for key,value in state.items() if key.startswith('projection:')})
        assert self.context.metadata()==definition['projection_context']
        with np.load(p['dataset'],allow_pickle=False) as z:
            carrier = torch.as_tensor(z['real_carrier384'][self.mean['REAL_FIT_rows']],dtype=torch.float64)
        real_cpu = observe_existing_carrier(carrier,q_floor=float(self.loss.q_floor),return_carrier=True)
        real = {key:value.to(self.device) if isinstance(value,torch.Tensor) else value for key,value in real_cpu.items()}
        real['flags'] = {key:value.to(self.device) for key,value in real_cpu['flags'].items()}
        self.real_wave = real['window_V']
        self.memory = GlobalDistributionMemory01(real,q_floor=float(self.loss.q_floor),**p['memory'])
        with np.load(p['source_memory'],allow_pickle=False) as z:
            self.saved = {key:z[key].copy() for key in z.files}
        np.testing.assert_allclose(self.memory.target.cpu().numpy(),self.saved['target'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(self.memory.joint.cholesky.cpu().numpy(),self.saved['joint_cholesky'],atol=1e-12,rtol=0)
        for key in ['target','weights','wave_denominator']:
            getattr(self.memory,key).copy_(torch.as_tensor(self.saved[key],device=self.device))
        self.memory.joint.cholesky.copy_(torch.as_tensor(self.saved['joint_cholesky'],device=self.device))
        self.supported = np.flatnonzero(self.saved['wave_supported'])
        self.progress('inputs_and_actual_memory_ready',SOURCE_FIT_parents=6465,REAL_FIT_rows=18948,parameters=163)

    def mean_delta(self, coefficients):
        head = coefficients[:161]*self.steps[:161]
        delta = np.empty_like(self.mean['original_mean'])
        for first in range(0,6465,128):
            last = min(first+128,6465)
            decoded = self.mean['decoded'][first:last].astype(np.float64)
            windows = np.lib.stride_tricks.sliding_window_view(np.pad(decoded,((0,0),(0,0),(2,2))),5,axis=2)
            logits = np.einsum('bctk,ck->bt',windows,head[:160].reshape(32,5))+head[160]
            raw0 = self.mean['raw_mean'][first:last]
            raw1 = raw0+.20*(np.tanh(self.mean['logits'][first:last]+logits)-np.tanh(self.mean['logits'][first:last]))*self.mean['gate']
            q0,q1 = raw0[:,110:366].sum(1),raw1[:,110:366].sum(1)
            assert np.all(q1>1e-6)
            qbar = self.mean['qbar'][first:last,None]
            delta[first:last] = qbar*raw1/q1[:,None]-qbar*raw0/q0[:,None]
        delta += self.radii[:,None]*np.einsum('d,dpt->pt',coefficients[161:]*self.steps[161:]*self.additive_physical_steps,self.fields)
        assert np.max(abs(delta[:,:100]))==0
        rms = np.sqrt(np.mean(delta**2,axis=1)/np.mean(self.mean['original_mean']**2,axis=1))
        return delta,float(rms.max())

    @torch.no_grad()
    def bank_metrics(self, means, bank):
        value = torch.as_tensor(means,device=self.device)
        memory = self.memory
        metrics = {key:float(v) for key,v in memory.losses(value,value).items()}
        selection = memory.sections['native_joint']
        raw = memory.joint.cholesky@(value-memory.target)[selection]
        variance = memory.joint.cholesky.square().sum(1)
        for name in ['wave','peak_codes']:
            s = memory.joint.block_slices[name]
            metrics['block_'+name] = float((raw[s].square()/variance[s]).mean())
        real_mass = memory.target[memory.sections['Q_mass']]
        mass = value[memory.sections['Q_mass']]
        s = memory.sections['PSD_CDF'];width=memory.threshold_count
        residual = real_mass[:,None]*(value[s].reshape(-1,width)/mass[:,None]-memory.target[s].reshape(-1,width)/real_mass[:,None])
        metrics['fixed_REAL_conditional_PSD'] = float((memory.weights[s].reshape(-1,width)*residual.square()).sum())
        local = memory.conditional_wave_risks(value,value).cpu().numpy()
        inherited = self.source['records']['inherited']['metrics_by_bank'][bank]
        reference = np.array([inherited[memory.wave_cell_names[j]] for j in self.supported])
        metrics['local_excess_penalty'] = float(np.mean(np.maximum(local[self.supported]/(1.02*reference)-1,0)**2))
        return metrics

    @torch.no_grad()
    def evaluate(self, coefficients, keep_arrays=False):
        started = time.monotonic()
        assert np.sum(abs(coefficients))<=1+1e-8
        delta,rms = self.mean_delta(coefficients)
        sums = np.zeros((2,len(self.memory.target)))
        flags = [Counter(),Counter()]
        wave = np.empty((103440,256))
        arrays = []
        for bank,take in self.segments:
            shift = delta[self.parent[take]]
            left,fraction = self.left[take].astype(np.int64),self.fraction[take]
            warped = np.take_along_axis(shift,left,axis=1)*(1-fraction)+np.take_along_axis(shift,left+1,axis=1)*fraction
            carrier = self.cache['pre_ADC_carrier0'][take]+warped*self.multiplier[take]
            obs = observe_generated_carrier(torch.as_tensor(carrier,device=self.device),q_floor=float(self.loss.q_floor),return_carrier=True)
            wave[take] = obs['window_V'].cpu().numpy()
            sums[bank] += self.memory.sums(obs).cpu().numpy()
            flags[bank].update({key:int(v.sum()) for key,v in obs['flags'].items()})
            if keep_arrays:
                arrays.append((take,dict(window_adc_codes=((obs['window_V']+obs['baseline_mean_V'][:,None]+.5)*16383).round().cpu().numpy().astype(np.uint16),prefix_adc_codes=((obs['baseline100']+.5)*16383).round().cpu().numpy().astype(np.uint16),baseline_mean_V=obs['baseline_mean_V'].cpu().numpy(),crop_start=obs['crop_start'].cpu().numpy(),**{'flag_'+key:v.cpu().numpy() for key,v in obs['flags'].items()})))
        sums /= 51720
        actual_risks = [self.bank_metrics(sums[bank],bank) for bank in range(2)]
        self.progress('actual_hard_guards_complete',cached_observation_rows=103440)
        fake = torch.as_tensor(wave,device=self.device)
        w,parts = complete_wave_value(fake,self.real_wave,self.loss,self.context,self.progress)
        q = float(self.loss._qpsd_with_context(fake,self.real_wave,None,self.context,False))
        assert self.loss.last_diagnostics['conditional_coverage_complete']
        record = dict(values=dict(wave=w,QPSD=q),wave_components=parts,metrics_by_bank=actual_risks,flags=[dict(x) for x in flags],mean_relative_RMS_max=rms,seconds=time.monotonic()-started,cached_observation_rows=103440)
        if keep_arrays:
            full = {key:np.empty((103440,)+value.shape[1:],dtype=value.dtype) for key,value in arrays[0][1].items()}
            for take,data in arrays:
                for key,value in data.items():full[key][take]=value
            record['readout'] = full
            record['memory_means'] = sums
        return record
