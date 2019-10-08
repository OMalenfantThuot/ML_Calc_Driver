import os
import pytest
import numpy as np
from mlcalcdriver import Posinp, Job
from mlcalcdriver.base import JobResults

pos_folder = "tests/posinp_files/"

class TestJob:

    file1, file2, file3 = os.path.join(pos_folder, "free.xyz"), os.path.join(pos_folder, "surface.xyz"), os.path.join(pos_folder, "N2.xyz")
    pos1, pos2, pos3 = Posinp.from_file(file1), Posinp.from_file(file2), Posinp.from_file(file3)

    job = Job(posinp=[pos1, pos2, pos3])

    def test_raises_no_positions(self):
        with pytest.raises(ValueError):
            j = Job()

    def test_raises_posinp_types(self):
        with pytest.raises(TypeError):
            j = Job(posinp=[self.pos1,1])

    def test_posinp_types(self):
        job1 = Job(posinp=self.pos1)
        job2 = Job(posinp=[self.pos1])
        assert job1.posinp == job2.posinp

    @pytest.mark.parametrize(
        "value, expected",
        [
            (job.num_struct, 3),
            (job.posinp[1], pos2),
            (job.results.positions, job.posinp),
            (job.results.n_at, [4, 4, 2])]
        )

    def test_values(self, value, expected):
        assert value == expected
