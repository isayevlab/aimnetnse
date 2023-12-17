from typing import Optional, Sequence, List

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class AIMNetNSECalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'charges']

    def __init__(self, model, charge: float = 0, mult: int = 1, **kwargs):
        super().__init__(*kwargs)
        self.model = model
        self.device = list(model.parameters())[0].device
        self.charge = charge
        # TODO: assume multiplicity from charge
        self.mult = mult if mult else abs(charge) + 1

    def calculate(self, atoms: Optional[Atoms] = None, properties: Sequence[str] = ('energy',),
                  system_changes: List[str] = all_changes):
        super().calculate(atoms, properties, system_changes)

        # initialize input for AIMNet
        coord = torch.from_numpy(self.atoms.positions).to(torch.float).to(self.device).unsqueeze(0)
        number_idx = torch.from_numpy(self.atoms.numbers).to(torch.float).to(self.device).unsqueeze(0)
        charge = torch.tensor([self.charge]).to(torch.float).to(self.device)
        mult = torch.tensor([self.mult]).to(torch.float).to(self.device)
        X = dict(coord=coord, numbers=number_idx, charge=charge, mult=mult)

        if 'forces' in properties:
            coord.requires_grad_(True)

        # calculate energies and charges
        Y = self.model(X)
        Y['charges_ab'] = Y['charges'].squeeze()
        Y['charges'] = Y['charges'].sum(-1)  # merge alpha and beta atomic charges

        # calculate forces
        if 'forces' in properties:
            e = Y['energy'].sum(-1)
            Y['forces'] = - torch.autograd.grad(e, coord, torch.ones_like(e))[0].squeeze()

        # convert to numpy arrays
        output = {k: v.detach().cpu().numpy()
                  for k, v in Y.items()}

        output['energy'] = output['energy'].sum(-1)
        output['charges'] = output['charges'].squeeze()

        # update charges
        for atom, charge in zip(self.atoms, output['charges']):
            atom.charge = charge

        # save results
        self.results.update(output)


class AIMNet2Calculator(Calculator):
    """ ASE calculator for AIMNet2 model
    Arguments:
        model (:class:`torch.nn.Module`): AIMNet2 model
        charge (int or float): molecular charge.  Default: 0
    """

    implemented_properties = ['energy', 'forces', 'free_energy', 'charges']

    def __init__(self, model, charge=0):
        super().__init__()
        self.model = model
        self.charge = charge
        self.device = next(model.parameters()).device
        cutoff = max(v.item() for k, v in model.state_dict().items() if k.endswith('aev.rc_s'))
        self.cutoff = float(cutoff)
        self._t_numbers = None
        self._t_charge = None

    def do_reset(self):
        self._t_numbers = None
        self._t_charge = None
        self.charge = 0.0

    def set_charge(self, charge):
        self.charge = float(charge)

    def _make_input(self):
        coord = torch.as_tensor(self.atoms.positions).to(torch.float).to(self.device).unsqueeze(0)
        if self._t_numbers is None:
            self._t_numbers = torch.as_tensor(self.atoms.numbers).to(torch.long).to(self.device).unsqueeze(0)
            self._t_charge = torch.tensor([self.charge], dtype=torch.float, device=self.device)
        d = dict(coord=coord, numbers=self._t_numbers, charge=self._t_charge)
        return d

    def _eval_model(self, d, forces=True):
        prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(forces)
        if forces:
            d['coord'].requires_grad_(True)
        _out = self.model(d)
        ret = dict(energy=_out['energy'].item(), charges=_out['charges'].detach()[0].cpu().numpy())
        if forces:
            if 'forces' in _out:
                f = _out['forces'][0]
            else:
                f = - torch.autograd.grad(_out['energy'], d['coord'])[0][0]
            ret['forces'] = f.detach().cpu().numpy()
        torch._C._set_grad_enabled(prev)
        return ret

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        _in = self._make_input()
        do_forces = 'forces' in properties
        _out =  self._eval_model(_in, do_forces)

        self.results['energy'] = _out['energy']
        self.results['charges'] = _out['charges']
        if do_forces:
            self.results['forces'] = _out['forces']

