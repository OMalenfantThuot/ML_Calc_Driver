import ase
import numpy as np
from mlcalcdriver.base import Posinp


def posinp_to_ase_atoms(posinp):
    r"""
    Converts a :class:`Posinp` instance to an :class:`ase.Atoms`
    instance.
    """
    symbols, positions, masses = "", [], []
    for atom in posinp.atoms:
        symbols += atom.type
        positions.append(atom.position)
        masses.append(atom.mass)
    pbc = [False if dim == 0.0 else True for dim in posinp.cell.lengths()]
    atoms = ase.Atoms(
        symbols=symbols, positions=positions, masses=masses, cell=posinp.cell, pbc=pbc
    )
    return atoms


def ase_atoms_to_posinp(atoms):
    r"""
    Converts an :class:`ase.Atoms` instance to a
    :class:`Posinp` instance.
    """
    pos_dict = {"units": "angstroem"}
    positions = []
    for at in atoms:
        positions.append({at.symbol: at.position})
    cell = atoms.get_cell()
    pos_dict["positions"] = positions
    pos_dict["cell"] = cell
    return Posinp.from_dict(pos_dict)
