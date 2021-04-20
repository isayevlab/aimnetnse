from eval import EnsembleCalculator, load_models, to_numpy
from openbabel import pybel
from tqdm.auto import tqdm
import os
import numpy as np


def guess_pybel_type(filename):
    assert '.' in filename
    return os.path.splitext(filename)[1][1:]


def calc_react_idx(data):
    ip = data['energy'][0] - data['energy'][1]
    ea = data['energy'][1] - data['energy'][2]
    f_el = data['charges'][1] - data['charges'][0]
    f_nuc = data['charges'][2] - data['charges'][1]
    chi = 0.5 * (ip + ea)
    eta = 0.5 * (ip - ea)
    omega = (chi ** 2) / (2 * eta)
    f_rad = 0.5 * (f_el + f_nuc)
    _omega = np.expand_dims(omega, axis=-1)
    omega_el = f_el * _omega
    omega_nuc = f_nuc * _omega
    omega_rad = f_rad * _omega
    return dict(ip=ip, ea=ea, f_el=f_el, f_nuc=f_nuc, f_rad=f_rad,
                chi=chi, eta=eta, omega=omega,
                omega_el=omega_el, omega_nuc=omega_nuc, omega_rad=omega_rad)
    
def mol2data(mol, device):
    coord = np.array([a.coords for a in mol.atoms])
    numbers = np.array([a.atomicnum for a in mol.atoms])
    coord = torch.tensor(coord, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1).to(device)
    numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0).repeat(3, 1).to(device)
    charge = torch.tensor([1, 0, -1]).to(device)  # cation, neutral, anion
    mult = torch.tensor([2, 1, 2]).to(device)
    return dict(coord=coord, numbers=numbers, charge=charge, mult=mult)


if __name__ == '__main__':
    import torch
    import argparse
    import ujson
    from tqdm.auto import tqdm
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--in-file', nargs='?', help='Multi-molecule input file. Extension should be an acceptable to OpnBabel file type.')
    parser.add_argument('--out', help='Output multi-line JSON file with computed properties.',
                        type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--allow-charged', action='store_true', default=False, 
                        help='Skip check for molecule neutral charge. Useful for reading XYZ files, e.g. when OpenBabel guess for molecular charge is wrong.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_models(args.models).to(device)

    in_format = guess_pybel_type(args.in_file)
    for mol in tqdm(pybel.readfile(in_format, args.in_file)):
        if not args.allow_charged and mol.charge != 0:
            raise Warning('Charged molecules are not accepted!')
            continue
        # create input: cation, neutral, anion
        data = mol2data(mol, device)
        # disable optimizations for safety. with some combinations of pytorch/cuda it's getting very slow
        with torch.jit.optimized_execution(False), torch.no_grad():       
            pred = model(data)
        pred['charges'] = pred['charges'].sum(-1)
        pred = to_numpy(pred)
        # calculate indicies
        pred.update(calc_react_idx(pred))
        # write
        for k, v in pred.items():
            pred[k] = v.tolist()
        ujson.dump(pred, args.out)
    for k, v in pred.items():
        print(k, np.array(v).shape)

