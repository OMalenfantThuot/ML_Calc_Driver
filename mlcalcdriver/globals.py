r"""
Defines some global values and parameters used in
classes and workflows of the package.
"""

ATOMS_MASS = {
    "H": 1.00794,
    "He": 4.002602,
    "Li": 6.941,
    "Be": 9.012182,
    "B": 10.811,
    "C": 12.011,
    "C12": 12,
    "C13": 13.003,
    "C14": 14.003,
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
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.933,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.39,
    "Ga": 69.732,
    "Ge": 72.61,
    "As": 74.922,
    "Se": 78.09,
    "Br": 79.904,
    "Kr": 84.80,
    "Rb": 84.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.94,
    "Tc": 98.907,
    "Ru": 101.07,
    "Rh": 102.906,
    "Pd": 106.42,
    "Ag": 107.868,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.71,
    "Sb": 121.760,
    "Te": 127.6,
    "I": 126.904,
    "Xe": 131.29,
    "Cs": 132.905,
    "Ba": 137.327,
    "La": 138.906,
    "Ce": 140.115,
    "Pr": 140.908,
    "Nd": 144.24,
    "Pm": 144.913,
    "Sm": 150.36,
    "Eu": 151.966,
    "Gd": 157.25,
    "Tb": 158.925,
    "Dy": 162.50,
    "Ho": 164.930,
    "Er": 167.26,
    "Tm": 168.934,
    "Yb": 173.04,
    "Lu": 174.967,
    "Hf": 178.49,
    "Ta": 180.948,
    "W": 183.85,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.22,
    "Pt": 195.08,
    "Au": 196.967,
    "Hg": 200.59,
    "Tl": 204.383,
    "Pb": 207.2,
    "Bi": 208.980,
    "Po": 208.982,
    "At": 209.987,
    "Rn": 222.018,
    "Fr": 223.020,
    "Ra": 226.025,
    "Ac": 227.028,
    "Th": 232.038,
    "Pa": 231.036,
    "U": 238.029,
    "Np": 237.048,
    "Pu": 244.064,
    "Am": 243.061,
    "Cm": 247.070,
}
r"""
`Dictionnary` containing elemental masses, in atomic mass units.
Used to compute vibrational energies.
"""

####
# Conversion factors
####

AMU_TO_EMU = 1.660538782e-27 / 9.10938215e-31
r"""
Conversion factor for atomic mass unit to electronic mass unit.
"""

EMU_TO_AMU = 1.0 / AMU_TO_EMU
r"""
Conversion factor for electronic mass unit to atomic mass unit.
"""

B_TO_ANG = 0.529177249
r"""
Conversion factor for bohr to angstrom.
"""

ANG_TO_B = 1.0 / B_TO_ANG
r"""
Conversion factor from angstrom to bohr.
"""

HA_TO_CMM1 = 219474.6313705
r"""
Conversion factor from Hartree to :math:`cm^{-1}`.
"""

HA_TO_EV = 27.21138602
r"""
Conversion factor for Hartree to eV.
"""

EV_TO_HA = 1 / HA_TO_EV
r"""
Conversion factor for eV to Hartree.
"""

DEBYE_TO_AU = 0.393430307
r"""
Conversion factor for Debye to atomic units of dipole moment.
"""

AU_TO_DEBYE = 1 / DEBYE_TO_AU
r"""
Conversion factor for atomic units of dipole moment to Debye.
"""

# Units dictionary

eVA = {"positions": "angstrom", "energy": "eV", "dipole_moment": "Debye"}
