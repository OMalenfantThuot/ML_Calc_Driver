import os
import pytest
from mlcalcdriver import Posinp
from mlcalcdriver.interfaces import posinp_to_ase_atoms, ase_atoms_to_posinp

pos_folder = "tests/posinp_files/"

class TestConversion:

    def test_mol(self):
        pos1 = Posinp.from_file(os.path.join(pos_folder, "N2.xyz"))
        atoms = posinp_to_ase_atoms(pos1)
        pos2 = ase_atoms_to_posinp(atoms)
        assert pos1 == pos2

    def test_surface(self):
        pos1 = Posinp.from_file(os.path.join(pos_folder, "surface2.xyz"))
        atoms = posinp_to_ase_atoms(pos1)
        pos2 = ase_atoms_to_posinp(atoms)
        assert pos1 == pos2

    def test_periodic(self):
        pos1 = Posinp.from_file(os.path.join(pos_folder, "periodic.xyz"))
        atoms = posinp_to_ase_atoms(pos1)
        pos2 = ase_atoms_to_posinp(atoms)
        assert pos1 == pos2
