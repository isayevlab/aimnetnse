from pathlib import Path
import torch
from torch import nn, Tensor
from typing import Dict


class EnsembleCalculator(nn.Module):
    def __init__(self, *models, out_keys=('energy', 'charges')):
        super().__init__()
        self.add_module('models', nn.ModuleList(models))
        self.out_keys = out_keys

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # alpha and beta total charge
        data['charge'] = 0.5 * torch.stack([data['charge'] - data['mult'] + 1, data['charge'] + data['mult'] - 1], dim=-1)
        outputs = dict((k, []) for k in self.out_keys)
        for model in self.models:
            o = model(data)
            for k in self.out_keys:
                outputs[k].append(o[k][-1])
        result = dict()
        for k in self.out_keys:
            v = torch.stack(outputs[k], dim=0)
            result[k] = v.mean(dim=0)
            result[k + '_std'] = v.std(dim=0)
        return result


def load_models(device: str = 'cpu'):
    aimnetnse_dir = Path(__file__).resolve().parent
    resources_path = aimnetnse_dir / 'resources'
    files = resources_path.glob('*.jpt')
    models = [torch.jit.load(f) for f in files]
    ensemble = EnsembleCalculator(*models)
    return ensemble.to(device)
