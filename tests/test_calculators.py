import os
import pytest
import numpy as np
from mlcalcdriver.calculators import Calculator, SchnetPackCalculator

model_folder = "tests/models/"

class TestCalc:

    mp1 = os.path.join(model_folder, "N2_model")

    def test_calc_needs_methods(self):
        with pytest.raises(NotImplementedError):
            c = Calculator()
        c = Calculator(available_properties="energy")
        assert(c.available_properties == "energy")
        with pytest.raises(NotImplementedError):
            c.run()
