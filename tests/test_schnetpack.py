import os
import pytest
import numpy as np
from mlcalcdriver import Posinp, Job
from mlcalcdriver.calculators import SchnetPackCalculator
from copy import deepcopy

pos_folder = "tests/posinp_files/"
model_folder = "tests/models/"


class TestSchnetPack:

    pos1 = Posinp.from_file(os.path.join(pos_folder, "N2.xyz"))
    model1 = os.path.join(model_folder, "ani1_N2_model")
    model2 = os.path.join(model_folder, "wacsf_model")
    model3 = os.path.join(model_folder, "H2O_model")
    calc1 = SchnetPackCalculator(model_dir=model1)
    calc2 = SchnetPackCalculator(model_dir=model2)
    calc3 = SchnetPackCalculator(model_dir=model3)
    job = Job(posinp=pos1, calculator=calc1)
    jobwacsf = Job(posinp=pos1, calculator=calc2)

    def test_only_energy(self):
        self.job.run("energy")
        assert np.array(-2979.6067) == self.job.results["energy"][0]
        new_job = deepcopy(self.job)
        new_job.calculator.units = {"energy": "hartree", "positions": "atomic"}
        new_job.run("energy")
        assert new_job.results["energy"][0] == np.array(-81079.228)
        self.job.results["energy"] = None

    def test_forces_from_deriv(self):
        assert self.job.calculator.available_properties == ["energy"]
        self.job.run("forces")
        ref_forces = np.array([[-0.0, -0.0, -0.31532133], [-0.0, -0.0, 0.31532133]])
        assert np.isclose(self.job.results["forces"][0], ref_forces).all()
        new_job = deepcopy(self.job)
        new_job.calculator.units = {"energy": "hartree", "positions": "atomic"}
        new_job.run("forces")
        assert np.isclose(ref_forces * 51.42206334724, new_job.results["forces"][0]).all()

    def test_forces_from_finite_difference(self):
        assert self.job.calculator.available_properties == ["energy"]
        self.job.run("forces", finite_difference=True)
        ref_forces = np.array([[-0.0, -0.0, -0.32416448], [-0.0, -0.0, 0.32416448]])
        assert np.isclose(self.job.results["forces"][0], ref_forces).all()

    def test_wacsf(self):
        self.jobwacsf.run("energy")
        assert self.jobwacsf.results["energy"] is not None

    def test_bad_property(self):
        with pytest.raises(ValueError):
            self.job.run("dipole")

    def test_convert(self):
        pos_angstrom = Posinp.from_file(os.path.join(pos_folder, "H2O_unrelaxed.xyz"))
        pos_atomic = Posinp.from_file(os.path.join(pos_folder, "H2O_atomic.xyz"))
        job1 = Job(posinp=pos_angstrom, calculator=self.calc3)
        job2 = Job(posinp=pos_atomic, calculator=self.calc3)
        job1.run("energy")
        job2.run("energy")
        assert job1.results["energy"] == job2.results["energy"]
