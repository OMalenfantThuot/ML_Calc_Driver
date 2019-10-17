import os
import pytest
import numpy as np
from mlcalcdriver import Posinp, Job
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.workflows import Geopt

pos_folder = "tests/posinp_files/"
model_folder = "tests/models/"


class TestGeopt:

    posN2 = Posinp.from_file(os.path.join(pos_folder, "N2_unrelaxed.xyz"))
    posH2O = Posinp.from_file(os.path.join(pos_folder, "H2O_unrelaxed.xyz"))
    calcN2 = SchnetPackCalculator(os.path.join(model_folder, "ani1_N2_model"))
    calcH2O = SchnetPackCalculator(os.path.join(model_folder, "H2O_model"))

    def test_N2(self, capsys):
        init = Job(posinp=self.posN2, calculator=self.calcN2)
        init.run("energy")
        assert init.results["energy"][0] == -2978.2354
        geo = Geopt(posinp=self.posN2, calculator=self.calcN2)
        geo.run(recenter=True, verbose=2)
        output = capsys.readouterr()
        assert 1.101 < geo.final_posinp.distance(0, 1) < 1.103
        assert np.isclose(
            geo.final_posinp.positions[0, 2],
            -geo.final_posinp.positions[1, 2],
            atol=1e-6,
        )
        assert output.out is not None
        final = Job(posinp=geo.final_posinp, calculator=self.calcN2)
        final.run("energy")
        assert final.results["energy"][0] == -2979.6070

    def test_H2O(self):
        init = Job(posinp=self.posH2O, calculator=self.calcH2O)
        init.run("energy")
        assert np.isclose(init.results["energy"][0], -2076.7576, atol=1e-4)
        geo = Geopt(posinp=self.posH2O, calculator=self.calcH2O)
        geo.run()
        assert 0.964 < geo.final_posinp.distance(0, 1) < 0.965
        assert (
            geo.final_posinp.distance(0, 1) - geo.final_posinp.distance(0, 2)
        ) < 0.0001
        # TODO Check the angle
        final = Job(posinp=geo.final_posinp, calculator=self.calcH2O)
        final.run("energy")
        assert np.isclose(final.results["energy"][0], -2078.6301, atol=1e-4)

    def test_errors(self):
        with pytest.raises(ValueError):
            geo = Geopt(posinp=None, calculator=self.calcN2)

        with pytest.raises(TypeError):
            geo = Geopt(posinp=self.posN2, calculator=1)
