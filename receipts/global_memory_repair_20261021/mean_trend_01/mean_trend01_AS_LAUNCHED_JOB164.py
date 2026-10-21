"""Train the diagnosed mean-conditioning path with exact streamed adjoints."""
import copy
import math
import torch

from .global_memory01 import MeanHeadRepair01
from .model import ResidualFiLM
from .model008 import canonical_charge008, relative_time_warp008


class ConditionedMeanRepair01(MeanHeadRepair01):
    def __init__(self, baseline, encoding, filtered, inherited):
        super().__init__(baseline, encoding, filtered)
        self.mean_head.load_state_dict(inherited.mean_head.state_dict())
        self.mean_time.load_state_dict(inherited.mean_time.state_dict())
        self.mean_time.requires_grad_(False)
        self.unet = copy.deepcopy(baseline.unet).requires_grad_(False)
        self.unet.condition.requires_grad_(True)
        for block in self.unet.modules():
            if isinstance(block, ResidualFiLM):
                block.film.requires_grad_(True)
        source = encoding['source']
        centered = source - source[:, :100].mean(1, keepdim=True)
        self.unit_source = centered / centered.amax(1, keepdim=True)
        self.features = baseline._features(self.unit_source[:, None])
        self.cached = None
        self.forward_rows = 0

    def unit_shape(self, features, filtered, scalars):
        self.forward_rows += len(features)
        decoded, _ = self.unet(features, scalars)
        residual = .20 * self.mean_head(decoded).tanh().double() * self.gate
        scale = torch.exp(math.log(2.) * self.mean_time(scalars).squeeze(1).tanh().double())
        warped, _ = relative_time_warp008(filtered, scale)
        raw = warped * self.gate + residual
        charge = canonical_charge008(raw)
        if bool((charge <= 1e-6).any()):
            raise ValueError('Conditioned mean has nonpositive canonical charge')
        return (raw / charge[..., None])[:, 0]

    def uncached(self, rows):
        unit = self.unit_shape(self.features[rows], self.filtered[rows], self.scalars[rows])
        return self.encoding['original']['qbar'][rows, None] * unit

    def forward(self, rows):
        return self.uncached(rows) if self.cached is None else self.cached[rows]

    def refresh(self, parents, chunk, *, grad=False):
        with torch.no_grad():
            values = self.encoding['original']['signal_pre'].detach().clone()
            for rows in parents.split(chunk):
                values[rows] = self.uncached(rows)
        self.cached = values.requires_grad_(grad)

    def map_adjoints(self, gradients, parents, chunk):
        """One replay per parent chunk; every draw's adjoint was summed first."""
        parameters = [v for v in self.parameters() if v.requires_grad]
        result = {key: [torch.zeros_like(v) for v in parameters] for key in gradients}
        self.cached = None
        for rows in parents.split(chunk):
            wave = self.uncached(rows)
            for index, (key, grad) in enumerate(gradients.items()):
                values = torch.autograd.grad(wave, parameters, grad_outputs=grad[rows],
                                             retain_graph=index + 1 < len(gradients))
                for total, value in zip(result[key], values):
                    total.add_(value)
        return result


class MeanTrendGrid01:
    """FIT shape prototypes, continuous log-Q grid, full-waveform curvature."""
    def __init__(self, mean, parents, *, centers=(.207919414, .312575162), points=128):
        source = mean.encoding['source']
        psd = source[:, 156:366].sum(1) / source[:, 110:366].sum(1)
        self.features, self.filtered, self.scalars, self.spacing, self.groups = [], [], [], [], []
        for center in centers:
            rows = parents[(psd[parents] - center).abs() < 1e-6]
            assert len(rows) >= points
            self.groups.append(rows)
            low, high = mean.scalars[rows, 0].min(), mean.scalars[rows, 0].max()
            logq = torch.linspace(low, high, points, device=source.device)
            scalars = mean.scalars[rows].mean(0).expand(points, -1).clone()
            # Scaling a fixed normalized shape changes logQ and log amplitude equally.
            scalars[:, 1] = logq + (mean.scalars[rows, 1] - mean.scalars[rows, 0]).mean()
            scalars[:, 0] = logq
            self.scalars.append(scalars)
            self.features.append(mean.features[rows].mean(0, keepdim=True).expand(points, -1, -1))
            self.filtered.append(mean.filtered[rows].mean(0, keepdim=True).expand(points, -1, -1))
            self.spacing.append((high - low).double() / (points - 1))

    def losses(self, mean):
        result = []
        for features, filtered, scalars, spacing in zip(self.features, self.filtered, self.scalars, self.spacing):
            wave = mean.unit_shape(features, filtered, scalars)
            second = (wave[2:] - 2 * wave[1:-1] + wave[:-2]) / spacing.square()
            result.append(second.square().mean())
        return torch.stack(result)

    def excess(self, mean):
        result = []
        for rows in self.groups:
            ordered = rows[mean.scalars[rows, 0].argsort(stable=True)]
            wave = mean(ordered)
            psd = wave[:, 156:366].sum(1) / wave[:, 110:366].sum(1)
            # numpy-style medians, including the average of the middle pair.
            medians = torch.stack([part.quantile(.5) for part in torch.tensor_split(psd, 128)])
            result.append(medians.diff().abs().sum() - (medians[-1] - medians[0]).abs())
        return torch.stack(result)
