import os
import pytest
import numpy as np
from ase.cell import Cell
from mlcalcdriver import Posinp
from mlcalcdriver.base import Atom

tests_fol = "tests/posinp_files/"


class TestPosinp:

    # Posinp with surface boundary conditions
    surface_filename = os.path.join(tests_fol, "surface.xyz")
    pos = Posinp.from_file(surface_filename)
    # Posinp with free boundary conditions
    free_filename = os.path.join(tests_fol, "free.xyz")
    free_pos = Posinp.from_file(free_filename)
    periodic_filename = os.path.join(tests_fol, "periodic.xyz")
    periodic_pos = Posinp.from_file(periodic_filename)
    # Posinp read from a string
    string = """\
4   atomic
free
C    0.6661284109   0.000000000   1.153768252
C    3.330642055    0.000000000   1.153768252
C    4.662898877    1.000000000   3.461304757
C    7.327412521    0.000000000   3.461304757"""
    str_pos = Posinp.from_string(string)

    value = [len(pos), pos.units, pos.boundary_conditions, pos.cell, pos[0], pos.angles]
    expected = [
        4,
        "reduced",
        "surface",
        Cell.new(np.array([8.07007483423, 0.0, 4.65925987792])),
        Atom("C", [0.08333333333, 0.5, 0.25]),
        np.array([90.0, 90.0, 90.0]),
    ]

    def test_from_file(self):
        for v, e in zip(self.value, self.expected):
            if isinstance(v, np.ndarray) or isinstance(v, Cell):
                assert np.allclose(v, e)
            else:
                assert v == e

    def test_from_string(self):
        assert self.str_pos == self.free_pos

    def test_repr(self):
        atoms = [Atom("C", [0, 0, 0]), Atom("N", [0, 0, 1])]
        new_pos = Posinp(atoms, units="angstrom", boundary_conditions="free")
        msg = (
            "Posinp([Atom('C', [0.0, 0.0, 0.0]), Atom('N', [0.0, 0.0, "
            "1.0])], 'angstrom', 'free', cell=Cell([0.0, 0.0, 0.0]), angles=[90. 90. 90.])"
        )
        assert repr(new_pos) == msg

    def test_write(self):
        fname = os.path.join(tests_fol, "test.xyz")
        self.pos.write(fname)
        assert self.pos == Posinp.from_file(fname)
        os.remove(fname)

    def test_free_boundary_conditions_has_no_cell(self):
        assert (self.free_pos.cell == Cell.new()).all()

    def test_translate_atom(self):
        new_pos = self.pos.translate_atom(0, [0.5, 0, 0])
        assert new_pos != self.pos
        assert new_pos[0] == Atom("C", [0.58333333333, 0.5, 0.25])

    @pytest.mark.parametrize(
        "fname", ["free_reduced.xyz", "missing_atom.xyz", "additional_atom.xyz"]
    )
    def test_init_raises_ValueError(self, fname):
        with pytest.raises(ValueError):
            Posinp.from_file(os.path.join(tests_fol, fname))

    @pytest.mark.parametrize(
        "to_evaluate",
        [
            "Posinp([Atom('C', [0, 0, 0])], 'bohr', 'periodic')",
            "Posinp([Atom('C', [0, 0, 0])], 'bohr', 'periodic', cell=[1, 1])",
            "Posinp([Atom('C', [0, 0, 0])], 'bohr', 'periodic', cell=[1,'inf',1])",
        ],
    )
    def test_init_raises_ValueError2(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)

    def test_positions(self):
        expected = [7.327412521, 0.0, 3.461304757]
        pos1 = Posinp(
            [Atom("C", expected)], units="angstrom", boundary_conditions="free"
        )
        pos2 = pos1.translate_atom(0, [-7.327412521, 0.0, -3.461304757])
        assert np.allclose(pos1.positions, expected)
        assert np.allclose(pos2.positions, [0, 0, 0])

    def test___eq__(self):
        atom1 = Atom("N", [0.0, 0.0, 0.0])
        atom2 = Atom("N", [0.0, 0.0, 1.1])
        pos1 = Posinp([atom1, atom2], "angstrom", "free")
        pos2 = Posinp([atom2, atom1], "angstrom", "free")
        assert pos1 == pos2  # The order of the atoms in the list do not count
        assert pos1 != 1  # No error if other object is not a posinp

    def test_with_surface_boundary_conditions(self):
        # Two Posinp instances with surface BC are the same even if they
        # have a different cell size along y-axis
        pos_with_inf = Posinp(
            [
                Atom(
                    "N",
                    [2.97630782434901e-23, 6.87220595204354e-23, 0.0107161998748779],
                ),
                Atom(
                    "N",
                    [-1.10434491945017e-23, -4.87342174483075e-23, 1.10427379608154],
                ),
            ],
            "angstrom",
            "surface",
            cell=[40, ".inf", 40],
        )
        with pytest.raises(ValueError):
            pos_wo_inf = Posinp(
                [
                    Atom(
                        "N",
                        [2.97630782434901e-23, 6.87220595204354e-23, 0.0107161998748779],
                    ),
                    Atom(
                        "N",
                        [-1.10434491945017e-23, -4.87342174483075e-23, 1.10427379608154],
                    ),
                ],
                "angstrom",
                "surface",
                cell=[40, 40, 40],
            )
        # They are obviously different if the cell size along the other
        # directions are not the same
        pos2_with_inf = Posinp(
            [
                Atom(
                    "N",
                    [2.97630782434901e-23, 6.87220595204354e-23, 0.0107161998748779],
                ),
                Atom(
                    "N",
                    [-1.10434491945017e-23, -4.87342174483075e-23, 1.10427379608154],
                ),
            ],
            "angstrom",
            "surface",
            cell=[20, "inf", 40],
        )
        assert pos_with_inf != pos2_with_inf
        # They still have the same BC
        assert pos2_with_inf.boundary_conditions == pos_with_inf.boundary_conditions

    def test_to_centroid(self):
        atoms = [Atom("N", [0, 0, 0]), Atom("N", [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstrom", boundary_conditions="free")
        expected_atoms = [Atom("N", [0, 0, -0.55]), Atom("N", [0, 0, 0.55])]
        expected_pos = Posinp(
            expected_atoms, units="angstrom", boundary_conditions="free"
        )
        assert pos.to_centroid() == expected_pos

    def test_to_barycenter(self):
        atoms = [Atom("N", [0, 0, 0]), Atom("N", [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstrom", boundary_conditions="free")
        expected_atoms = [Atom("N", [0, 0, -0.55]), Atom("N", [0, 0, 0.55])]
        expected_pos = Posinp(
            expected_atoms, units="angstrom", boundary_conditions="free"
        )
        assert pos.to_barycenter() == expected_pos

    def test_distance(self):
        assert np.isclose(self.free_pos.distance(0, 2), 4.722170992308181)

    def test_convert(self):
        pos1 = self.periodic_pos.positions
        assert self.periodic_pos.units == "angstrom"
        self.periodic_pos.convert_units("atomic")
        assert np.isclose(
            self.periodic_pos.positions,
            np.array(
                [
                    [0.15747717, 0.94486299, 0.4724315],
                    [0.78738583, 0.94486299, 0.4724315],
                    [1.10234016, 0.94486299, 1.41729449],
                    [1.73224882, 0.94486299, 1.41729449],
                ]
            ),
        ).all()
        assert self.periodic_pos.units == "atomic"
        self.periodic_pos.convert_units("angstrom")
        assert np.isclose(self.periodic_pos.positions, pos1).all()
        assert self.periodic_pos.units == "angstrom"
        self.periodic_pos.convert_units("angstrom")
        assert np.isclose(self.periodic_pos.positions, pos1).all()
        red = Posinp.from_file(tests_fol + "reduced.xyz")
        red.convert_units("angstrom")
        print(red.positions)
        assert np.isclose(
            red.positions,
            np.array(
                [
                    [0.4233418, 1.32294312, 0.95251905],
                    [3.81007619, 0.26458862, 0.47625952],
                ]
            ),
        ).all()

    def test_angles(self):
        h2o = Posinp.from_file(tests_fol + "H2Orelaxed.xyz")
        a = h2o.angle(1, 0, 2) * 180 / np.pi
        assert np.isclose(a, 104.1219, atol=10 ** -4)
        a1, a2 = h2o.angle(0, 1, 2), h2o.angle(2, 1, 0)
        assert a1 == a2
