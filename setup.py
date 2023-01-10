"""Install nglib dependencies and register it globally.

For installing test dependencies run: python3 setup.py test

"""

from setuptools import setup, find_packages

setup(name='tgpy',
      version='0.1.0',
      description='Transport Gaussian Process in Python',
      author='Gonzalo Rios',
      author_email='gonzalo@riosycia.cl',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'matplotlib', 'seaborn', 'torch', 'dill', 'tqdm',
                        'seaborn', 'mpl-scatter-density', 'utm', 'pydeck'],
      zip_safe=False)
