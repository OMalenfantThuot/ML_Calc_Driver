import os
import pytest
import numpy as np
from mlcalcdriver import Posinp, Job
from mlcalcdriver.calculators import SchnetPackCalculator

pos_folder = "tests/posinp_files/"
model_folder = "tests/models/"


class TestSchnetPack:

    pos1 = Posinp.from_file(os.path.join(pos_folder, "N2.xyz"))
    model1 = os.path.join(model_folder, "N2_model")
    calc1 = SchnetPackCalculator(model_dir=model1)

    def test_only_energy(self):
        job = Job(posinp=self.pos1, calculator=self.calc1)
        job.run("energy")
        assert np.array(-2979.6067) == job.results["energy"][0]

    def test_forces_from_deriv(self):
        job = Job(posinp=self.pos1, calculator=self.calc1)
        assert job.calculator.available_properties == ["energy"]
        job.run("forces")
        assert np.float32(-2979.6067) == job.results["energy"][0]
        ref_forces = np.array([[-0.0, -0.0, -0.32416448], [-0.0, -0.0, 0.32416448]])
        assert np.isclose(job.results["forces"][0], ref_forces).all()
