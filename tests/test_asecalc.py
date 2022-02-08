import os
import pytest
import numpy as np
from ase.io import read
from mlcalcdriver.calculators import AseSpkCalculator, AseEnsembleCalculator
from copy import deepcopy

pos_folder = "tests/posinp_files/"
model_folder = "tests/models/"


class TestAseCalc:

    atoms = read(os.path.join(pos_folder, "H2O_unrelaxed.cif"))
    model1 = os.path.join(model_folder, "H2O_forces_model")
    model2 = os.path.join(model_folder, "H2O_model")
    calc1 = AseSpkCalculator(model1, md=True)
    calc2 = AseSpkCalculator(model2)

    def test_model1(self):
        at = deepcopy(self.atoms)
        at.calc = deepcopy(self.calc1)
        f1 = at.get_forces()
        e1 = at.get_total_energy()
        assert np.isclose(e1, -477.344909, atol=1e-04)
        ref_forces = np.array(
            [
                [0, 0.7281927, -9.122646],
                [0, -3.1064584, 4.289081],
                [0, 2.3782659, 4.8335648],
            ],
            np.float32,
        )
        assert np.isclose(f1, ref_forces, atol=1e-04).all()

    def test_model2(self):
        at = deepcopy(self.atoms)
        at.calc = self.calc2
        e2 = at.get_total_energy()
        assert np.isclose(e2, -2076.7575, atol=1e-04)
        f2 = at.get_forces()
        ref_forces = np.array(
            [
                [0, 0.80131567, -6.6086245],
                [0, -1.4727035, 3.3589983],
                [0, 0.6713877, 3.2496262],
            ],
            np.float32,
        )
        assert np.isclose(f2, ref_forces, atol=1e-04).all()

    def test_both_models(self):
        at = deepcopy(self.atoms)
        calc = AseEnsembleCalculator(modelpaths=[self.model1, self.model2])
        at.calc = calc
        _ = at.get_forces()
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
        assert np.isclose(at.calc.results["forces"], ref_forces, atol=1e-04).all()
        assert np.isclose(at.calc.results["forces_std"], ref_std, atol=1e-04).all()
