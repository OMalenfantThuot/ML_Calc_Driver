import ase

def posinp_to_ase_atoms(posinp):
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
