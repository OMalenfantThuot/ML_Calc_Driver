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
    if posinp.boundary_conditions == "free":
        pbc = False
    elif posinp.boundary_conditions == "surface":
        pbc = (True, False, True)
    elif posinp.boundary_conditions == "periodic":
        pbc = True
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
    if cell.orthorhombic:
        if (cell == 0.0).all():
            new_cell = None
        elif cell[1, 1] in [0.0, np.inf]:
            new_cell = [cell[0, 0], str(np.inf), cell[2, 2]]
        else:
            new_cell = [dim[i] for i, dim in enumerate(cell)]
    else:
        raise NotImplementedError("Non orthorhombic cells are not supported yet.")
    pos_dict["positions"] = positions
    pos_dict["cell"] = new_cell
    return Posinp.from_dict(pos_dict)
