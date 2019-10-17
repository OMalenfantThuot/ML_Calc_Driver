r"""
Defines some global values and parameters used in 
classes and workflows of the package.
"""

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

####
# Conversion factors
####

# Conversion from atomic to electronic mass unit
AMU_TO_EMU = 1.660538782e-27 / 9.10938215e-31
# Conversion from electronic to atomic mass unit
EMU_TO_AMU = 1.0 / AMU_TO_EMU
# Conversion factor from bohr to angstroem
B_TO_ANG = 0.529177249
# Conversion factor from angstroem to bohr
ANG_TO_B = 1.0 / B_TO_ANG
# Conversion factor from Hartree to cm^-1
HA_TO_CMM1 = 219474.6313705
# Conversion factor from Hartree to electron-Volt
HA_TO_EV = 27.21138602
# Conversion factor from electron-Volt to Hartree
EV_TO_HA = 1 / HA_TO_EV
# Conversion factor from Debye to atomic units of dipole moment
DEBYE_TO_AU = 0.393430307
# Conversion factor from atomic units of dipole moment to Debye
AU_TO_DEBYE = 1 / DEBYE_TO_AU
