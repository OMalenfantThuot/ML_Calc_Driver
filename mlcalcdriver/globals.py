r"""
Defines some global values and parameters used in 
classes and workflows of the package.
"""

__all__ = ["ATOMS_MASS"]

# Mass of the different types of atoms in atomic mass units
# TODO: Add more types of atoms
# (found in $SRC_DIR/bigdft/src/orbitals/eleconf-inc.f90)
ATOMS_MASS = {
    "H": 1.00794,
    "He": 4.002602,
    "Li": 6.941,
    "Be": 9.012182,
    "B": 10.811,
    "C": 12.011,
    "N": 14.00674,
    "O": 15.9994,
    "F": 18.9984032,
    "Ne": 20.1797,
    "Na": 22.989768,
    "Mg": 24.3050,
    "Al": 26.981539,
    "Si": 28.0855,
    "P": 30.973762,
    "S": 32.066,
    "Cl": 35.4527,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955910,
    "Ti": 47.88,
    "V": 50.9415,
}
