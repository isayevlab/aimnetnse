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
            Y['forces'] = - torch.autograd.grad(e, coord, torch.ones_like(e))[0]

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
