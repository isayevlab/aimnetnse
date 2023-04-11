#!/usr/bin/env python
import torch
from ase.io import Trajectory, write
from aimnetnse import AIMNetNSECalculator, load_models
from ase.build import molecule
from ase.optimize import BFGS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_models(device)

atoms = molecule('H2O')
atoms.info['charge'] = 0
atoms.info['mult'] = 1
atoms.calc = AIMNetNSECalculator(model)
atoms.get_potential_energy()

# write output to xyz
with open('example1.xyz', 'w') as traj:
    opt = BFGS(atoms)
    opt.attach(atoms.write, interval=1, filename=traj)
    opt.run(fmax=1e-4)
