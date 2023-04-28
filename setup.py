from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='aimnetnse',
    description='AIMNet-NSE: Prediction of energies and spin-polarized charges with neural network potential',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/isayevlab/aimnetnse',
    author='Roman Zubatyuk',
    author_email='zubatyuk@gmail.com',
    license='https://choosealicense.com/',
    packages=find_packages(),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'ase',
        'torch',
        'tqdm',
        'h5py',
        'importlib_metadata',
    ]
)