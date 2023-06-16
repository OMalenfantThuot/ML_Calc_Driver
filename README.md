# Machine Learning Calculations Driver (ML\_Calc\_Driver)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Build Status](https://travis-ci.org/OMalenfantThuot/ML_Calc_Driver.svg?branch=master&kill_cache=1)](https://travis-ci.org/OMalenfantThuot/ML_Calc_Driver)
[![Coverage Status](https://coveralls.io/repos/github/OMalenfantThuot/ML_Calc_Driver/badge.svg?branch=master&kill_cache=1)](https://coveralls.io/github/OMalenfantThuot/ML_Calc_Driver)
[![PyPi](https://img.shields.io/pypi/v/mlcalcdriver.svg?kill_cache=1)](https://pypi.org/project/mlcalcdriver/)
[![python](https://img.shields.io/pypi/pyversions/mlcalcdriver.svg)](https://www.python.org/)
[![PyUp](https://pyup.io/repos/github/OMalenfantThuot/ML_Calc_Driver/shield.svg?kill_cache=1)](https://pyup.io/repos/github/OMalenfantThuot/ML_Calc_Driver)

ML\_Calc\_Driver is a driver to emulate DFT calculations using machine learned predictive models.
To use this package, one needs an already trained model, that can predict energy or forces from an input atomic geometry.
Supported systems depends on the model, this package should not be a limitation.

This package is tested with python 3.10.

Credit to [mmoriniere](https://gitlab.com/mmoriniere) for the [MyBigDFT package](https://gitlab.com/mmoriniere/MyBigDFT)
which served as a foundation for this work. Some classes have been directly adapted from MyBigDFT.

**Dependencies scanned by [PyUp.io](https://pyup.io/)**

## Documentation

The documentation is available [here](https://omalenfantthuot.github.io/ML_Calc_Driver/).

## Installation

### From PyPi

To install the latest version available on PyPi:

`pip install mlcalcdriver`

To upgrade to last version:

`pip install -U mlcalcdriver`

### From sources

```
git clone git@github.com:OMalenfantThuot/ML_Calc_Driver.git
cd ML_Calc_Driver/
pip install -r requirements_dev.txt
pip install .
```
To modify the sources, instead of `pip install .`, use

```
pip install -e .
```
