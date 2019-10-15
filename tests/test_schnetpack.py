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
    job = Job(posinp=pos1, calculator=calc1)

    def test_only_energy(self):
        self.job.run("energy")
        assert np.array(-2979.6067) == self.job.results["energy"][0]

    def test_cuda_unavailable(self):
        with pytest.warns(UserWarning):
            self.job.run("energy", device="cuda")

    def test_forces_from_deriv(self):
        assert self.job.calculator.available_properties == ["energy"]
        self.job.run("forces")
        assert np.float32(-2979.6067) == self.job.results["energy"][0]
        ref_forces = np.array([[-0.0, -0.0, -0.32416448], [-0.0, -0.0, 0.32416448]])
        assert np.isclose(self.job.results["forces"][0], ref_forces).all()
        assert self.job.results.atom_types == [{"N"}]
        assert self.job.results.boundary_conditions == ["free"]
        assert self.job.results.cell == [None]

    def test_bad_property(self):
        with pytest.raises(ValueError):
            self.job.run("dipole")
