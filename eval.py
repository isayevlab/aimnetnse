import torch
from torch import nn, Tensor
from typing import Dict
import numpy as np
from tqdm.auto import tqdm


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


def load_models(files):
    models = [torch.jit.load(f) for f in files]
    return EnsembleCalculator(*models)


def to_device(data, device):
    for k in data:
        data[k] = data[k].to(device)
    return data

def to_tensors(data):
    for k in data:
        data[k] = torch.tensor(data[k])
    return data

def to_numpy(data):
    for k in data:
        if isinstance(data[k], Tensor):
            data[k] = data[k].detach().cpu().numpy()
    return data


def eval_batched(model, data, batchsize=128, in_keys=('coord', 'numbers', 'charge', 'mult'), output_device='cpu'):
    assert set(data.keys()).issubset(in_keys), 'Not all the required data are provided!'
    device = next(model.parameters()).device
    data = to_device(data, device)
    l = data[in_keys[0]].shape[0]
    batch_idx = torch.arange(l, device=device).split(batchsize)
    outputs = list()
    for idx in tqdm(batch_idx, leave=False):
        o = model(dict((k, v[idx]) for k, v in data.items()))
        outputs.append(to_device(o, output_device))
    ret = dict()
    for k in outputs[0].keys():
        ret[k] = torch.cat([o[k] for o in outputs], dim=0)
    return ret


def _delta_prp(batch, charges=(0, 1), props=('energy', 'charges')):
    mol_ids = batch['mol_id']
    charge = batch['charge'] if batch['charge'].ndim == 1 else batch['charge'].sum(-1)
    w_chg0 = charge == charges[0]
    w_chg1 = charge == charges[1]
    mol_ids, idx0, idx1 = np.intersect1d(mol_ids[w_chg0], mol_ids[w_chg1], return_indices=True)
    return [np.array(batch[p][w_chg0][idx0] - batch[p][w_chg1][idx1]) for p in props], mol_ids


def calc_prp(batch, energy_key='energy', charges_key='charges'):
    ret = dict()
    ret['mol_id'] = np.unique(batch['mol_id'])
    m, n = ret['mol_id'].shape[0], batch[charges_key].shape[1]
    for k in ('ip', 'ea', 'chi', 'eta', 'omega'):
        ret[k] = np.zeros((m, )) * np.nan
    for k in ('f_el', 'f_nuc', 'f_rad', 'omega_el', 'omega_nuc', 'omega_rad'):
        ret[k] = np.zeros((m, n)) * np.nan

    (ip, f_el), ids_ip = _delta_prp(batch, charges=(1, 0), props=(energy_key, charges_key))
    f_el = - f_el
    _, idx0, idx1 = np.intersect1d(ret['mol_id'], ids_ip, return_indices=True)
    ret['ip'][idx0] = ip[idx1]
    ret['f_el'][idx0] = - f_el[idx1]

    (ea, f_nuc), ids_ea = _delta_prp(batch, charges=(0, -1), props=(energy_key, charges_key))
    _, idx0, idx1 = np.intersect1d(ret['mol_id'], ids_ea, return_indices=True)
    ret['ea'][idx0] = ea[idx1]
    ret['f_nuc'][idx0] = - f_nuc[idx1]

    ret['chi'] = 0.5 * (ret['ip'] + ret['ea'])
    ret['eta'] = 0.5 * (ret['ip'] - ret['ea'])
    ret['omega'] = (ret['chi'] ** 2) / (2 * ret['eta'])
    ret['f_rad'] = 0.5 * (ret['f_el'] + ret['f_nuc'])
    ret['omega_el'] = ret['f_el'] * ret['omega'][:, None]
    ret['omega_nuc'] = ret['f_nuc'] * ret['omega'][:, None]
    ret['omega_rad'] = ret['f_rad'] * ret['omega'][:, None]
    return ret


def rmse(x, y):
    return ((x - y) ** 2).mean() ** 0.5


def mad(x, y):
    return np.abs(x - y).mean()


def merge_prp(prp_true, prp_pred, keys):
    d = dict()
    w = ~np.isnan(prp_true[keys[0]])
    ids, idx0, idx1 = np.intersect1d(prp_pred['mol_id'], prp_true['mol_id'][w], return_indices=True)
    for k in keys:
        d[k] = prp_pred[k][idx0]
        d[k + '_pred'] = prp_true[k][w][idx1]
    d['mol_id'] = ids
    return d


def print_errors(data, keys):
    for key in keys: 
        _true = np.concatenate([d[key].flatten() for d in data])
        _pred = np.concatenate([d[key + '_pred'].flatten() for d in data])
        _rmse, _mad = rmse(_true, _pred), mad(_true, _pred)
        print(f'{key}: Data points {len(_true)} RMSE {_rmse:.3g} MAD {_mad:.3g}')


if __name__ == '__main__':
    import h5py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('h5file')
    parser.add_argument('models', nargs='+')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_models(args.models).to(device)

    # properties for all molecules (energy, charges)
    data_all = list()
    # properties for neutral-anion pairs (ea, f_nuc)
    data_ea = list()
    # properties for neutral-cation pairs (ip, f_el)
    data_ip = list()
    # properties for neutral-cation-anion (chi, eta, omega, f_rad, omega_el, omega_nuc, omega_rad)
    data_ea_ip = list()

    # iterate groups in data file
    _n = 0
    with h5py.File(args.h5file, 'r') as f:
        for group_name in tqdm(f):
            g = f[group_name]
            _n += g['numbers'].shape[0]
            data = dict((k, v[()]) for k, v in g.items())
            data['charges'] = data['charges'].sum(-1)  # merge alpha and beta atomic charges

            # make predictions
            # input should be Dict[str, Tensor] containing these keys:
            X = dict((k, data[k]) for k in ('coord', 'numbers', 'charge', 'mult'))
            X = to_tensors(X)
            
            # disable optimizations for safety. with some combinations of pytorch/cuda it's getting very slow
            with torch.jit.optimized_execution(False), torch.no_grad():
                Y = eval_batched(model, X)
            Y['charges'] = Y['charges'].sum(-1)  # merge alpha and beta atomic charges
            Y = to_numpy(Y)
            Y['mol_id'] = data['mol_id']
            Y['charge'] = data['charge']
            # save predictions with '_pred' suffix 
            for k in ('energy', 'charges'):
                data[k + '_pred'] = Y[k]

            data_all.append(data)

            # compute ip, ea, etc.
            prp_true = calc_prp(data)
            prp_pred = calc_prp(Y)

            # since not all molecules (conformers) have all three charge states successfully calculated, 
            # will match by mol_id and write only those

            data_ea.append(merge_prp(prp_true, prp_pred, ('ea', 'f_nuc')))
            data_ip.append(merge_prp(prp_true, prp_pred, ('ip', 'f_el')))
            data_ea_ip.append(merge_prp(prp_true, prp_pred, ('chi', 'eta', 'omega', 'f_rad', 'omega_el', 'omega_nuc', 'omega_rad')))

    print('Prediction errors [eV and e-]:')
    print_errors(data_all, ('energy', 'charges'))
    print_errors(data_ea, ('ea', 'f_nuc'))
    print_errors(data_ip, ('ip', 'f_el'))
    print_errors(data_ea_ip, ('chi', 'eta', 'omega', 'f_rad', 'omega_el', 'omega_nuc', 'omega_rad'))
    
