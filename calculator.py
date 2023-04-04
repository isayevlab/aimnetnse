import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from torch import Tensor


class AIMNetNSECalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'charges']

    def __init__(self, model, **kwargs):
        super().__init__(*kwargs)
        self.model = model
        self.device = list(model.parameters())[0].device

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # get electronic state information from atoms.info
        charge = self.atoms.info.get('charge', 0)
        # TODO: assume multiplicity from charge
        mult = atoms.info.get('mult', charge + 1)

        # initialize input for AIMNet
        coord = torch.from_numpy(self.atoms.positions).to(torch.float).to(self.device).unsqueeze(0)
        number_idx = torch.from_numpy(self.atoms.numbers).to(torch.float).to(self.device).unsqueeze(0)
        charge = torch.tensor([charge]).to(torch.float).to(self.device)
        mult = torch.tensor([mult]).to(torch.float).to(self.device)
        X = dict(coord=coord, numbers=number_idx, charge=charge, mult=mult)

        # TODO: test forces
        if 'forces' in properties:
            coord.requires_grad_(True)

        # disable optimizations for safety. with some combinations of pytorch/cuda it's getting very slow
        with torch.jit.optimized_execution(False), torch.no_grad():
            Y = self.model(X)
        Y['charges'] = Y['charges'].sum(-1)  # merge alpha and beta atomic charges
        # convert to numpy arrays
        Y.update({k: v.detach().cpu().numpy()
                  for k, v in Y.items()
                  if isinstance(v, Tensor)})
        output = Y.copy()

        # calculate forces
        if 'forces' in properties:
            e = output['energy'].sum(-1)
            output['forces'] = - torch.autograd.grad(e, coord, torch.ones_like(e))[0]

        output['energy'] = output['energy'].sum(-1)

        # convert to eV
        for prop in self.implemented_properties:
            if prop in output:
                output[prop] *= units.Hartree

        # update charges
        for atom, charge in zip(self.atoms, output['charges']):
            atom.charge = charge

        # save results
        self.results.update(output)
