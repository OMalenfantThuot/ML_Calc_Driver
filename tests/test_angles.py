import os
import pytest
import numpy as np
from copy import deepcopy
from ase.cell import Cell
from mlcalcdriver import Posinp, Job
from mlcalcdriver.calculators import SchnetPackCalculator

pos_folder = "tests/posinp_angles/"
model_folder = "tests/models/"


class TestGrapheneAngles:

    graphene2_name = os.path.join(pos_folder, "gra2.xyz")
    pos_2at = Posinp.from_file(graphene2_name)

    graphene4_name = os.path.join(pos_folder, "gra4_red.xyz")
    pos_4at_red = Posinp.from_file(graphene4_name)

    calc = SchnetPackCalculator(os.path.join(model_folder, "H2O_model"))

    def test_same_result(self):
        j1 = Job(posinp=self.pos_2at, calculator=self.calc)
        j1.run("energy")
        e1 = j1.results["energy"]

        j2 = Job(posinp=self.pos_4at_red, calculator=self.calc)
        j2.run("energy")
        e2 = j2.results["energy"]

        pos_4at = deepcopy(self.pos_4at_red)
        pos_4at.convert_units("angstrom")
        j3 = Job(posinp=pos_4at, calculator=self.calc)
        j3.run("energy")
        e3 = j3.results["energy"]

        assert e2 == e3
        assert np.isclose(e2, 2 * e1, atol=0.0002)

    def test_cell_with_angles(self):
        assert np.isclose(
            self.pos_2at.cell.array,
            np.array([[2.46, 0.0, 0.0], [0.0, 0.0, 0.0], [1.23, 0.0, 2.13042249]]),
        ).all()

    def test_verify_values(self):

        assert np.isclose(self.pos_2at.angles, np.array([90.0, 60.0, 90.0])).all()
        assert self.pos_4at_red.orthorhombic
        assert not self.pos_2at.orthorhombic
        assert np.isclose(
            self.pos_2at.cell.lengths(), np.array([2.46, 0.0, 2.46])
        ).all()
