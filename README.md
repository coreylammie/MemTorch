# MemTorch
![](https://img.shields.io/badge/license-GPL-blue.svg)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3760695.svg)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/memtorch/community)
![](https://readthedocs.org/projects/pip/badge/?version=latest)
![](https://api.travis-ci.org/coreylammie/MemTorch.svg)

MemTorch is a *Simulation Framework for Memristive Deep Learning Systems* which integrates directly with the well-known *PyTorch* Machine Learning (ML) library, which is presented in *MemTorch: An Open-source Simulation Framework for Memristive Deep Learning Systems*.

## Installation
 To install MemTorch from source:

```
git clone https://github.com/coreylammie/MemTorch
cd MemTorch
python setup.py install
```

*If CUDA is True in setup.py, CUDA Toolkit 10.1 and Microsoft Visual C++ Build Tools are required. If CUDA is False in setup.py, Microsoft Visual C++ Build Tools are required.*

Alternatively, MemTorch can be installed using the *pip* package-management system:

```
pip install memtorch # Supports CUDA and normal operation
pip install memtorch-cpu # Supports normal operation
```

## API & Example Usage
A complete API is avaliable [here](https://memtorch.readthedocs.io/). To learn how to use MemTorch, and to reproduce results of ‘*MemTorch: An Open-source Simulation Framework for Memristive Deep Learning Systems*’, we provide numerous Jupyter notebooks [here](memtorch/examples).

## Current Issues and Feature Requests
Current issues, feature requests and improvements are welcome, and are tracked using: https://github.com/coreylammie/MemTorch/projects/1.

These should be reported [here](https://github.com/coreylammie/MemTorch/issues).

## Contributing
Please follow the "fork-and-pull" Git workflow:
 1. **Fork** the repo on GitHub
 2. **Clone** the project to your own machine
 3. **Commit** changes to your own branch
 4. **Push** your work back up to your fork
 5. Submit a **Pull request** so that we can review your changes

*Be sure to merge the latest from 'upstream' before making a pull request*.

## Citation

To cite MemTorch, kindly use the following BibTex entry:

```
@software{corey_lammie_2020_3760696,
  author       = {Corey Lammie and
                  Wei Xiang and
                  Bernabé Linares-Barranco and
                  Mostafa Rahimi Azghadi},
  title        = {{coreylammie/MemTorch: Initial Release}},
  month        = Apr.,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.3760695},
  url          = {https://doi.org/10.5281/zenodo.3760696}
}
```

## License
All code is licensed under the GNU General Public License v3.0. Details pertaining to this are available at: https://www.gnu.org/licenses/gpl-3.0.en.html.

[![HitCount](http://hits.dwyl.io/coreylammie/MemTorch.svg)](http://hits.dwyl.io/coreylammie/MemTorch)
