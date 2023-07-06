import os
import pytest
import numpy as np
from mlcalcdriver import Posinp, Job
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.workflows.phonon import Phonon, PhononFromHessian

pos_folder = "tests/posinp_files/"
model_folder = "tests/models/"


class TestPhononFinite:
    posN2 = Posinp.from_file(os.path.join(pos_folder, "N2_unrelaxed.xyz"))
    calcN2 = SchnetPackCalculator(os.path.join(model_folder, "myN2_model"))

    def test_ph_N2(self):
        ph1 = Phonon(posinp=self.posN2, calculator=self.calcN2, finite_difference=True)
        ph1.run(batch_size=1)
        assert np.isclose(ph1.energies.max(), 2339.53, atol=0.01)
        assert all(np.abs(np.delete(ph1.energies, np.argmax(ph1.energies))) < 30)

        ph2 = Phonon(
            posinp=ph1._ground_state,
            calculator=ph1.calculator,
            relax=False,
            finite_difference=True,
            translation_amplitudes=0.03,
        )
        ph2.run()
        assert np.allclose(ph1.energies, ph2.energies)

        ph3 = Phonon(
            posinp=self.posN2,
            calculator=self.calcN2,
            finite_difference=True,
            relax=False,
        )
        ph3.run()
        assert not np.allclose(ph3.energies, ph1.energies, atol=100)

    def test_phonon_posinp_error(self):
        with pytest.raises(TypeError):
            ph = Phonon(posinp=None, calculator=self.calcN2)

    def test_phonon_calc_error(self):
        with pytest.raises(TypeError):
            ph = Phonon(posinp=self.posN2, calculator=None)


class TestPhononAutoGrad:
    posH2O = Posinp.from_file(os.path.join(pos_folder, "H2Orelaxed.xyz"))
    calc_ener = SchnetPackCalculator(os.path.join(model_folder, "H2O_model"))
    calc_for = SchnetPackCalculator(os.path.join(model_folder, "H2O_forces_model"))

    def test_ph_h2o_autograd_2nd_derivative(self):
        ph1 = Phonon(posinp=self.posH2O, calculator=self.calc_ener)
        ph1.run()
        ph1.energies.sort()
        assert np.allclose(ph1.energies[6:9], [1726, 3856, 3942], atol=1)

    def test_ph_h2o_autograd_1st_derivative(self):
        ph1 = Phonon(posinp=self.posH2O, calculator=self.calc_for)
        ph1.run()
        ph1.energies.sort()
        assert np.allclose(ph1.energies[6:9], [1589, 3703, 3812], atol=1)

    def test_ph_from_hessian(self):
        job = Job(posinp=self.posH2O, calculator=self.calc_for)
        job.run("hessian")
        ph = PhononFromHessian(posinp=self.posH2O, hessian=job.results["hessian"])
        ph.run()
        ph.energies.sort()
        assert np.allclose(ph.energies[6:9], [1589, 3705, 3814], atol=1)
