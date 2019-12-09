import os
import pytest
import numpy as np
from mlcalcdriver import Posinp, Job
from mlcalcdriver.base import JobResults
from mlcalcdriver.calculators import Calculator

pos_folder = "tests/posinp_files/"


class TestJob:

    file1, file2, file3 = (
        os.path.join(pos_folder, "free.xyz"),
        os.path.join(pos_folder, "surface.xyz"),
        os.path.join(pos_folder, "N2.xyz"),
    )
    pos1, pos2, pos3 = (
        Posinp.from_file(file1),
        Posinp.from_file(file2),
        Posinp.from_file(file3),
    )
    dummy = Calculator(
        available_properties="", units={"positions": "atomic", "energy": "eV"}
    )
    badCalc = dict()
    job = Job(name="test", posinp=[pos1, pos2, pos3], calculator=dummy)

    def test_raises_no_positions(self):
        with pytest.raises(ValueError):
            j = Job(calculator=self.dummy)

    def test_raises_posinp_types(self):
        with pytest.raises(TypeError):
            j = Job(posinp=[self.pos1, 1], calculator=self.dummy)

    def test_raises_bad_calc(self):
        with pytest.raises(TypeError):
            j = Job(posinp=[self.pos1], calculator=self.badCalc)

    def test_posinp_types(self):
        job1 = Job(posinp=self.pos1, calculator=self.dummy)
        job2 = Job(posinp=[self.pos1], calculator=self.dummy)
        assert job1.posinp == job2.posinp

    @pytest.mark.parametrize(
        "value, expected",
        [(job.name, "test"), (job.num_struct, 3), (job.posinp[1], pos2)],
    )
    def test_values(self, value, expected):
        assert value == expected
