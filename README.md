# AIMNet-NSE: Prediction of energies and spin-polarized charges with neural network potential


This repository contains supplementary data and code for the manuscript 

**"Teaching a neural network to attach and detach electrons from molecules"** by
Roman Zubatyuk, Justin S. Smith, Benjamin T. Nebgen, Sergei Tretiak, Olexandr Isayev
https://www.nature.com/articles/s41467-021-24904-0


## Models

The models directory contains JIT-compiled Pytorch AIMNet-NSE trained models . Five models were trained on 80/20 cross-validation splits of the training dataset. It is advised to use average prediction of these 5 models to get the most accurate results. The model was trained for neutral and ion-radical states of non-equilibrium conformations of organic molecules containing {H, C, N, O, F, Si, P, S, Cl} elements. Given molecular conformation and charge state, it predicts PBE0/ma-def2-SVP energies and NBO spin-polarized partial charges, as well as derived properties, such as ionization potential, electron affinity, Fukui functions, electronegativity, hardness, etc.

The models could be loaded with the `torch.jit.load` function. As an input, they accept dingle argument of type `Dict[str, Tensor]` with following data:

```
coords: shape (m, n, 3) - atomic coordinates in Angstrom 
numbers: shape (m, n) - atomic numbers
charge: shape (m, 2) - total alpha and beta molecular charges
```

For the convenience, `eval.py` script has a function to convert charge and multiplicity to the total alpha and beta molecular charges as: `ab_charges = 0.5 * torch.stack([charge - mult + 1, charge + mult - 1])`

## Test datasets

The Ions-16 and ChEMBL-20 datasets are available at http://doi.org/10.5281/zenodo.5007980

The datasets contain PBE0/ma-Def2-SVP energies and NBO atomic charges for the non-equilibrium conformers of neutral organic molecules randomly sampled from PubChem database (Ions-16) and B97-3c optimized conformations of  neutral organic molecules randomly sampled from ChEMBL database (ChEMBL-20). The number in the dataset name corresponds to the maximum number of non-hydrogen atoms in the molecules. The dataset which was used for training the AIMNet-NSE model contains molecules up to 12 non-H atoms, whereis ons-16 and ChEMBL-20 contain the molecules with 13 non-hydrogen atoms or more.

The datasets formatted as HDF5 files. Data group names have format as `_???`, where `???` corresponds to the number of atoms in molecules.  Each group contain data for `M` molecules, each having `N` atoms. The groups contain following datasets:

| Name    | Data type | Shape   | Description                    |
| :---    | :---      | :---    | :---                           |
| mol_id  | S24       | M       | Molecule ID                    |
| coord   | float32   | M, N, 3 | Cartesian coordinates, &#8491; |
| numbers | uint8     | M, N    | Atomic numbers                 |
| charge  | int8      | M       | Molecular charge               |
| mult    | uint8     | M       | Spin multiplicity              |
| energy  | float64   | M       | PBE0/ma-def2-SVP energy, eV    |
| charges | float32   | M, N, 2 | &#945; and &#946; NBO charges  |

The molecule ID is a hash of molecular conformation. In each group, there are up to 3 entries with the same `mol_id` value, but with different `charge`. Those correspond to neutral, cation-radical and anion-radical states.

Test datasets could be evaluated with AIMNet-NSE model wth `eval.py` script:

`python eval.py test_datasets/chembl20.h5 models/aimnet-nse-cv?.jpt`

## Inference script

```
usage: eval_mols.py [-h] [--models MODELS [MODELS ...]] [--in-file [IN_FILE]]
                    [--out OUT] [--allow-charged]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS [MODELS ...]
  --in-file [IN_FILE]   Multi-molecule input file. Extension should be an
                        acceptable to OpenBabel file type.
  --out OUT             Output multi-line JSON file with computed properties.
  --allow-charged       Skip check for molecule neutral charge. Useful for
                        reading XYZ files, e.g. when OpenBabel guess for
                        molecular charge is wrong.
```
The script reads several files with compiled models and constructs an ensembled AIMNet-NSE model. For each molecule in the `in-file` it calculates a set of properties and writes a json-formatted dict to the `out` file (stdout by default). The output keys are the following:
`energy, charges, ip, ea, f_el, f_nuc, f_rad, chi, eta, omega, omega_el, omega_nuc, omega_rad`. 
The units are eV and e. 

