"""Independent literal-array reference for synthetic011 tests, no NN or I/O.

This is not a scientific runner. The caller must supply explicit synthetic
arrays/statistics/projection banks. No original dataset or checkpoint is loaded.
"""
import numpy as np


def literal_w1(x, y):
    """Uniform empirical W1, exact merged CDF support, including unequal N/ties."""
    a, b = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if a.ndim != 2 or b.ndim != 2 or min(len(a), len(b)) == 0 or a.shape[1] != b.shape[1]:
        raise ValueError('Bad literal arrays')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Nonfinite literal arrays')
    answers = []
    for k in range(a.shape[1]):
        av, bv = np.sort(a[:, k]), np.sort(b[:, k])
        z = np.sort(np.concatenate((av, bv)))
        ca = np.searchsorted(av, z[:-1], side='right')/len(av)
        cb = np.searchsorted(bv, z[:-1], side='right')/len(bv)
        answers.append(np.sum(np.abs(ca-cb)*np.diff(z)))
    return float(np.mean(answers))


def expand_indices(row_multiplicity):
    counts = np.asarray(row_multiplicity)
    if counts.ndim != 1 or not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)) or counts.sum() == 0:
        raise ValueError('Not positive-total integer occurrences')
    return np.repeat(np.arange(len(counts)), counts.astype(np.int64))


def wave_views(y, floor):
    shape = y/np.maximum(y.max(1, keepdims=True), floor)
    views = dict(voltage=y, shape=shape, difference=np.diff(shape, axis=1),
                 curvature=np.diff(shape, n=2, axis=1))
    for size in (5, 17):
        pad = np.pad(shape, ((0, 0), (size//2, size//2)), mode='edge')
        smooth = np.lib.stride_tricks.sliding_window_view(pad, size, axis=1).mean(2)
        views['difference_smooth_'+str(size)] = np.diff(smooth, axis=1)
    return views


def qpsd_features(y, stats):
    q, floor, reference = y.sum(1), float(stats['q_floor']), float(stats['q_reference'])
    log = np.where(q >= floor, np.log(np.maximum(q, floor)/reference),
                   np.log(floor/reference)+np.arcsinh((q-floor)/floor))
    psd = y[:, 46:].sum(1)/(np.where(q < 0, -1., 1.)*np.maximum(np.abs(q), floor))
    return np.stack((log, psd), 1)


def literal_loss(fake, real, stats, banks):
    """Evaluate the documented full loss on explicitly expanded synthetic rows."""
    fake, real = np.asarray(fake, np.float64), np.asarray(real, np.float64)
    if fake.shape[1] != 256 or real.shape[1] != 256:
        raise ValueError('Use synthetic256 arrays')
    fv, rv = wave_views(fake, stats['peak_floor']), wave_views(real, stats['peak_floor'])
    f, r = {}, {}
    for name in fv:
        center, scale = np.asarray(stats['views'][name]['center']), np.asarray(stats['views'][name]['scale'])
        f[name], r[name] = (fv[name]-center)/scale, (rv[name]-center)/scale
    swd = float(np.mean([literal_w1(f[name]@banks['wave:'+name], r[name]@banks['wave:'+name]) for name in fv]))
    coordinate = float(np.mean([literal_w1(f[name], r[name]) for name in ('voltage', 'shape')]))
    joint = []
    for name in ('voltage', 'shape'):
        for lag in (1, 2, 4, 8):
            direction = banks[f'lag:{name}:{lag}']
            project = lambda x:(x[:, :-lag, None]*direction[0]+x[:, lag:, None]*direction[1]).reshape(len(x), -1)
            joint.append(literal_w1(project(f[name]), project(r[name])))
    short_lag = float(np.mean(joint))
    fq, rq = qpsd_features(fake, stats), qpsd_features(real, stats)
    center, scale = np.asarray(stats['qpsd']['center']), np.asarray(stats['qpsd']['scale'])
    fz, rz = (fq-center)/scale, (rq-center)/scale
    global_term = .5*(literal_w1(fz, rz)+literal_w1(fz@banks['qpsd'], rz@banks['qpsd']))
    fb = np.searchsorted(stats['charge_bin_boundaries'], fq[:, 0], side='left')
    rb = np.searchsorted(stats['charge_bin_boundaries'], rq[:, 0], side='left')
    counts, conditional = [], []
    for k in range(8):
        fm, rm = fb == k, rb == k
        supported = min(int(fm.sum()), int(rm.sum())) >= int(stats['minimum_conditional_support'])
        counts.append(dict(bin=k, fake_count=int(fm.sum()), real_count=int(rm.sum()), supported=supported))
        if supported:
            conditional.append(literal_w1(fz[fm, 1], rz[rm, 1]))
    qpsd = .5*(global_term+np.mean(conditional)) if conditional else global_term
    return dict(wave=.5*swd+.25*coordinate+.25*short_lag, QPSD=float(qpsd),
                swd=swd, coordinate=coordinate, short_lag=short_lag,
                conditional_counts=counts, conditional_bins_used=len(conditional),
                full_coverage=len(conditional) == 8)


def artificial_fixture(seed=145011):
    """Fresh analytic pulses and arbitrary synthetic frozen scales; no fit/data."""
    rng = np.random.default_rng(seed)
    t = np.arange(256)
    template = np.exp(-((t-16)/6)**2)+.15*np.exp(-np.maximum(t-16, 0)/28)
    rows = 36
    amplitudes = np.linspace(.01, .12, rows)
    fake = amplitudes[:, None]*template+2e-5*rng.normal(size=(rows, 256))
    real = amplitudes[:29, None]*(template*.93)+3e-5*rng.normal(size=(29, 256))
    sizes = dict(voltage=256, shape=256, difference=255, curvature=254,
                 difference_smooth_5=255, difference_smooth_17=255)
    directions = lambda d,n:(lambda x:x/np.linalg.norm(x, axis=0, keepdims=True))(rng.normal(size=(d,n)))
    banks = {'wave:'+name:directions(size, 32) for name,size in sizes.items()}
    for name in ('voltage', 'shape'):
        for lag in (1, 2, 4, 8):
            banks[f'lag:{name}:{lag}'] = directions(2, 4)
    banks['qpsd'] = directions(2, 16)
    boundaries = np.linspace(-3, 3, 7)
    stats = dict(fit_split='TRAIN', q_reference=1., q_floor=1e-4, peak_floor=1e-4,
        charge_bin_boundaries=boundaries, minimum_conditional_support=16,
        views={name:dict(center=np.zeros(size),scale=np.ones(size)) for name,size in sizes.items()},
        qpsd=dict(center=np.zeros(2),scale=np.ones(2)))
    return fake, real, stats, banks
