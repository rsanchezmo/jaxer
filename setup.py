from setuptools import setup, find_packages

# Leer las dependencias del archivo requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='jaxer',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=required
)
