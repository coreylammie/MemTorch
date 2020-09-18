<h1 align="center">
  <br>
  MemTorch
  <br>
</h1>

[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/)
![](https://img.shields.io/badge/license-GPL-blue.svg)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3760695.svg)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/memtorch/community)
![](https://readthedocs.org/projects/pip/badge/?version=latest)
![](https://api.travis-ci.org/coreylammie/MemTorch.svg)
[![codecov](https://codecov.io/gh/coreylammie/MemTorch/branch/master/graph/badge.svg)](https://codecov.io/gh/coreylammie/MemTorch)

MemTorch is a *Simulation Framework for Memristive Deep Learning Systems* which integrates directly with the well-known *PyTorch* Machine Learning (ML) library, which is presented in *MemTorch: An Open-source Simulation Framework for Memristive Deep Learning Systems*, which has been released [here](https://arxiv.org/abs/2004.10971).

<p align="center">
  <img src="Animate.gif" width=100%>
  <br>
  <em>Example usage.</em>
</p>

## MemTorch: An Open-source Simulation Framework for Memristive Deep Learning Systems
> Corey Lammie, Wei Xiang, Bernabé Linares-Barranco, and Mostafa Rahimi Azghadi<br>
>
> **Abstract:** *Memristive devices have shown great promise to facilitate the acceleration and improve the power efficiency of Deep Learning (DL) systems. Crossbar architectures constructed using memristive devices can be used to efficiently implement various in-memory computing operations, such as Multiply-Accumulate (MAC) and unrolled-convolutions, which are used extensively in Deep Neural Networks (DNNs) and Convolutional Neural Networks (CNNs). Currently, there is a lack of a modernized, open source and general high-level simulation platform that can fully integrate any behavioral or experimental memristive device model and its putative non-idealities into crossbar architectures within DL systems. This paper presents such a framework, entitled MemTorch, which adopts a modernized software engineering methodology and integrates directly with the well-known PyTorch Machine Learning (ML) library. We fully detail the public release of MemTorch and its release management, and use it to perform novel simulations of memristive DL systems, which are trained and benchmarked using the CIFAR-10 dataset. Moreover, we present a case study, in which MemTorch is used to simulate a near-sensor in-memory computing system for seizure detection using Pt/Hf/Ti Resistive Random Access Memory (ReRAM) devices. Our open source MemTorch framework can be used and expanded upon by circuit and system designers to conveniently perform customized large-scale memristive DL simulations taking into account various unavoidable device non-idealities, as a preliminary step before circuit-level realization.*

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
pip install memtorch-cpu # Supports normal operation
pip install memtorch # Supports CUDA and normal operation
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

To cite *MemTorch: An Open-source Simulation Framework for Memristive Deep Learning Systems*, use the following BibTex entry:

```
@misc{lammie2020memtorch,
  title={{MemTorch: An Open-source Simulation Framework for Memristive Deep Learning Systems}},
  author={Corey Lammie and Wei Xiang and Bernab\'e Linares-Barranco and Mostafa Rahimi Azghadi},
  month=Apr.,
  year={2020},
  eprint={2004.10971},
  archivePrefix={arXiv},
  primaryClass={cs.ET}
}
```

To cite this repository, use the following BibTex entry:

```
@software{corey_lammie_2020_3760696,
  author={Corey Lammie and Wei Xiang and Bernab\'e Linares-Barranco and Mostafa Rahimi Azghadi},
  title={{coreylammie/MemTorch: Initial Release}},
  month=Apr.,
  year={2020},
  publisher={Zenodo},
  doi={10.5281/zenodo.3760695},
  url={https://doi.org/10.5281/zenodo.3760696}
}
```

## License
All code is licensed under the GNU General Public License v3.0. Details pertaining to this are available at: https://www.gnu.org/licenses/gpl-3.0.en.html.

[![HitCount](http://hits.dwyl.io/coreylammie/MemTorch.svg)](http://hits.dwyl.io/coreylammie/MemTorch)
