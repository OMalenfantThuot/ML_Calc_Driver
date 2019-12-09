import os
import pytest
import numpy as np
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
