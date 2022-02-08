import os
import pytest
import numpy as np
from mlcalcdriver.globals import eVA
from mlcalcdriver.calculators import Calculator


class TestCalc:
    def test_calc_needs_methods(self):
        with pytest.raises(NotImplementedError):
            c = Calculator()
        with pytest.raises(NotImplementedError):
            c = Calculator(available_properties="energy")
        c = Calculator(available_properties="energy", units={"positions": "atomic"})
        assert c.available_properties == "energy"
        with pytest.raises(NotImplementedError):
            c.run()

    def test_units(self):
        with pytest.raises(TypeError):
            c = Calculator(available_properties="energy", units="atomic")
        with pytest.raises(KeyError):
            c = Calculator(available_properties="energy", units={"atomic": True})
        c = Calculator(available_properties="energy", units=eVA)
        assert c.units["positions"] == "angstrom"
        assert c.units["energy"] == "eV"
