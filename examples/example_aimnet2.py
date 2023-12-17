import torch
from aimnetnse import AIMNet2Calculator, load_models
from ase.build import molecule
from ase.optimize import LBFGS

model = load_models(version='2')
calc = AIMNet2Calculator(model)

atoms = molecule('C60')

charge = atoms.info.get('charge', 0)
calc.set_charge(charge)

atoms.calc = calc

atoms.get_potential_energy()

atoms.write('C60.xyz')

with torch.jit.optimized_execution(False):
    opt = LBFGS(atoms, trajectory='C60_opt.xyz', logfile='C60_opt.log')
    opt.run(fmax=5e-3, steps=2000)
