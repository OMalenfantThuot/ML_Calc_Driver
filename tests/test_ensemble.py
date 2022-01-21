import os
import pytest
import numpy as np
from mlcalcdriver import Posinp, Job
from mlcalcdriver.calculators import EnsembleCalculator
from copy import deepcopy

pos_folder = "tests/posinp_files/"
model_folder = "tests/models/"


class TestEnsemble:

    pos1 = Posinp.from_file(os.path.join(pos_folder, "H2O_unrelaxed.xyz"))
    model1 = os.path.join(model_folder, "H2O_forces_model")
    model2 = os.path.join(model_folder, "H2O_model")

    def test_both_models(self):
        ens = EnsembleCalculator(modelpaths=[self.model1, self.model2])
        job1 = Job(posinp=self.pos1, calculator=ens)
        job1.run("energy")
        assert job1.results["energy"] == np.array(-1277.0513)
        assert job1.results["energy_std"] == np.array(799.70636)

        ref_forces = np.array(
            [
                [0, 0.7647542, -7.8656354],
                [0, -2.2895808, 3.8240397],
                [0, 1.5248268, 4.0415955],
            ],
            dtype=np.float32,
        )
        ref_std = np.array(
            [
                [0, 0.03656149, 1.2570109],
                [0, 0.8168775, 0.4650414],
                [0, 0.8534391, 0.7919693],
            ],
            dtype=np.float32,
        )
        job2 = Job(posinp=self.pos1, calculator=ens)
        job2.run("forces")
        assert (job2.results["forces"] == ref_forces).all()
        assert (job2.results["forces_std"] == ref_std).all()
