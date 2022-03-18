r"""
To use a given type of machine learning model in conjunction with
ML_Calc_Driver, a related Calculator object must be defined.
If the class needed does not exist, the :class:`Calculator` class
must be used as a base for it. The `run()` and
`_get_available_properties()` methods must be defined, similarly
to the :class:`SchnetPackCalculator` class.
"""
from .calculator import Calculator
from .schnetpack import SchnetPackCalculator
from .ensemble import Ensemble, EnsembleCalculator, AseEnsembleCalculator
from .ase_calculators.asespkcalculator import AseSpkCalculator
from .schnetpack_patch import PatchSPCalculator
