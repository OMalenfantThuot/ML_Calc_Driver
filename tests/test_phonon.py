import os
import pytest
import numpy as np
from mlcalcdriver import Posinp, Job
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.workflows import Phonon

pos_folder = "tests/posinp_files/"
model_folder = "tests/models/"


class TestPhonon:

    posN2 = Posinp.from_file(os.path.join(pos_folder, "N2_unrelaxed.xyz"))
    calcN2 = SchnetPackCalculator(os.path.join(model_folder, "myN2_model"))

    def test_ph_N2(self):
        ph1 = Phonon(posinp=self.posN2, calculator=self.calcN2)
        ph1.run()
        assert np.isclose(ph1.energies.max(), 2339.57, atol=0.01)
        assert all(np.abs(ph1.energies[1:] < 30))

        ph2 = Phonon(
            posinp=ph1._ground_state,
            calculator=ph1.calculator,
            relax=False,
            translation_amplitudes=0.03,
        )
        ph2.run()
        assert np.allclose(ph1.energies, ph2.energies)

        ph3 = Phonon(posinp=self.posN2, calculator=self.calcN2, relax=False)
        ph3.run()
        assert not np.allclose(ph3.energies, ph1.energies, atol=100)

    def test_phonon_posinp_error(self):
        with pytest.raises(TypeError):
            ph = Phonon(posinp=None, calculator=self.calcN2)

    def test_phonon_calc_error(self):
        with pytest.raises(TypeError):
            ph = Phonon(posinp=self.posN2, calculator=None)

    # TODO verifiy normal modes
