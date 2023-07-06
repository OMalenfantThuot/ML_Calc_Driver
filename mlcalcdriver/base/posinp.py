r"""
The :class:`Posinp` and :class:`Atom` classes are represent the
atomic systems used as input for a calculation.
"""

from copy import deepcopy
from collections import Counter
from collections.abc import Sequence
import numpy as np
from ase.cell import Cell
from mlcalcdriver.globals import ATOMS_MASS, B_TO_ANG, ANG_TO_B
from mlcalcdriver.interfaces import ase_atoms_to_pos_dict


__all__ = ["Posinp", "Atom"]


class Posinp(Sequence):
    r"""
    Class allowing to initialize, read, write and interact with the
    input atomic geometry of a calculation in the form of an xyz file.

    Such a file is made of a few lines, containing all the necessary
    information to specify a given system of interest:

    * the first line contains the number of atoms :math:`n_{at}` and the
      units for the coordinates (and possibly the cell size),
    * the second line contains the boundary conditions used and possibly
      the simulation cell size (for periodic or surface boundary
      conditions),
    * the subsequent :math:`n_{at}` lines are used to define each atom
      of the system: first its type, then its position given by three
      coordinates (for :math:`x`, :math:`y` and :math:`z`).
    """

    def __init__(
        self,
        atoms,
        units="angstrom",
        boundary_conditions="free",
        cell=None,
        angles=None,
    ):
        r"""
        Parameters
        ----------
        atoms : list
            List of :class:`Atom` instances.
        units : str
            Units of the coordinate system.
        boundary_conditions : str
            Boundary conditions.
        cell : Sequence of length 3 or None
            Size of the simulation domain in the three space
            coordinates.


        >>> posinp = Posinp([Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])],
        ...                 'angstrom', 'free')
        >>> len(posinp)
        2
        >>> posinp.boundary_conditions
        'free'
        >>> posinp.units
        'angstrom'
        >>> for atom in posinp:
        ...     repr(atom)
        "Atom('N', [0.0, 0.0, 0.0])"
        "Atom('N', [0.0, 0.0, 1.1])"
        >>> posinp.masses
        array([14.00674, 14.00674])
        """
        # Check initial values
        # Set the base attributes
        self.atoms = atoms
        self.units = units
        self.boundary_conditions = boundary_conditions
        self.cell = (cell, angles)
        self.orthorhombic = self.cell.orthorhombic
        self._check_initial_values(self.cell, self.units, self.boundary_conditions)

    @staticmethod
    def _check_initial_values(cell, units, boundary_conditions):
        r"""
        Raises
        ------
        ValueError
            If some initial values are invalid, contradictory or
            missing.
        """
        lengths_counter = Counter(list(cell.lengths()))
        if boundary_conditions == "periodic" and lengths_counter[0] != 0:
            raise ValueError(
                """
                All 3 lattice vectors should have a non zero
                length in periodic boundary conditions.
                """
            )
        elif boundary_conditions == "surface" and lengths_counter[0] != 1:
            raise ValueError(
                "One dimension of the cell should be zero for a surface calculation."
            )
        elif boundary_conditions == "free" and lengths_counter[0] != 3:
            raise ValueError("All 3 lattice vectors should be 0 for free calculations.")
        if boundary_conditions == "free" and units == "reduced":
            raise ValueError("Cannot use reduced units with free boundary conditions")

    @classmethod
    def from_file(cls, filename):
        r"""
        Initialize the input positions from a file on disk.

        Parameters
        ----------
        filename : str
            Name of the input positions file on disk.

        Returns
        -------
        Posinp
            Posinp read from a file on disk.


        >>> posinp = Posinp.from_file("tests/surface.xyz")
        >>> posinp.cell
        [8.07007483423, 'inf', 4.65925987792]
        >>> print(posinp)
        4   reduced
        surface   8.07007483423   inf   4.65925987792
        C   0.08333333333   0.5   0.25
        C   0.41666666666   0.5   0.25
        C   0.58333333333   0.5   0.75
        C   0.91666666666   0.5   0.75
        <BLANKLINE>
        """
        with open(filename, "r") as stream:
            lines = [line.split() for line in stream.readlines()]
            return cls._from_lines(lines)

    @classmethod
    def read(cls, filename):
        r"""
        Initialize the input positions from a file on disk.
        Uses ase.io.read to support many file formats.

        Parameters
        ----------
        filename : str
            Name of the input file

        Returns
        -------
        Posinp
        """
        from ase.io import read

        return Posinp.from_ase(read(filename))

    @classmethod
    def from_string(cls, posinp):
        r"""
        Initialize the input positions from a string.

        Parameters
        ----------
        posinp : str
            Content of an xyz file as a string.

        Returns
        -------
        Posinp
            Posinp read from the string.
        """
        lines = [line.split() for line in posinp.split("\n")]
        lines = [line for line in lines if line]  # Remove empty lines
        return cls._from_lines(lines)

    @classmethod
    def _from_lines(cls, lines):
        r"""
        Initialize the input positions from a list of lines that mimics
        an xyz file.

        Parameters
        ----------
        lines : list
            List of the lines of the xyz file, each line being a list.

        Returns
        -------
        Posinp
            Posinp read from the list of lines.
        """
        # Decode the first line
        line1 = lines.pop(0)
        n_at = int(line1[0])
        units = line1[1]
        # Decode the second line
        line2 = lines.pop(0)
        boundary_conditions = line2[0].lower()
        if boundary_conditions == "free":
            cell = None
        else:
            cell = line2[1:4]
        # Angles if present
        if lines[0][0] == "angles":
            angles = np.array([float(a) for a in lines.pop(0)[1:]])
        else:
            angles = np.array([90.0, 90.0, 90.0])
        orthorhombic = True if (angles == 90.0).all() else False
        # Remove the lines about the forces, if there are some
        first_elements = [line[0] for line in lines]
        if "forces" in first_elements:
            index = first_elements.index("forces")
            lines = lines[:index]
        # Check the number of atoms is correct
        if n_at != len(lines):
            raise ValueError(
                "The number of atoms received is different from the expected "
                "number of atoms ({} != {})".format(len(lines), n_at)
            )
        # Decode the atoms
        atoms = []
        for line in lines:
            atom_type = line[0]
            position = np.array(line[1:4], dtype=float)
            isotope = line[4] if len(line) > 4 else None
            atoms.append(Atom(atom_type, position, isotope=isotope))
        return cls(atoms, units, boundary_conditions, cell=cell, angles=angles)

    @classmethod
    def from_dict(cls, posinp):
        r"""
        Initialize the input positions from a dictionary.

        Parameters
        ----------
        posinp : dict
            Posinp as a dictionary coming from an InputParams or
            Logfile instance.

        Returns
        -------
        Posinp
            Posinp initialized from an dictionary.


        >>> pos_dict = {
        ...     "units": "reduced",
        ...     "cell": [8.07007483423, 'inf', 4.65925987792],
        ...     "positions": [
        ...         {'C': [0.08333333333, 0.5, 0.25]},
        ...         {'C': [0.41666666666, 0.5, 0.25]},
        ...         {'C': [0.58333333333, 0.5, 0.75]},
        ...         {'C': [0.91666666666, 0.5, 0.75]},
        ...     ]
        ... }
        >>> pos = Posinp.from_dict(pos_dict)
        >>> pos.boundary_conditions
        'surface'

        If there is no "cell" key, then the boundary conditions are set
        to "free". Here, given that the units are reduced, this raises
        a ValueError:

        >>> del pos_dict["cell"]
        >>> pos = Posinp.from_dict(pos_dict)
        Traceback (most recent call last):
        ...
        ValueError: Cannot use reduced units with free boundary conditions
        """
        # Read data from the dictionary
        atoms = []  # atomic positions
        for atom in posinp["positions"]:
            atoms.append(Atom.from_dict(atom))
        units = posinp.get("units", "atomic")  # Units of the coordinates
        cell = posinp.get("cell")  # Simulation cell size
        angles = posinp.get("angles")
        # Infer the boundary conditions from the value of cell
        if cell is None:
            boundary_conditions = "free"
        else:
            if not isinstance(cell, Cell):
                cell = [
                    abs(float(size)) if size not in [".inf", "inf"] else 0.0
                    for size in cell
                ]
                cell = Cell.new(cell)
            lengths_counter = Counter(list(cell.lengths()))
            if lengths_counter[0] == 1:
                boundary_conditions = "surface"
            elif lengths_counter[0] == 3:
                boundary_conditions = "free"
            else:
                boundary_conditions = "periodic"
        return cls(atoms, units, boundary_conditions, cell=cell, angles=angles)

    @classmethod
    def from_ase(cls, atoms):
        r"""
        Parameters
        ----------
        atoms : ase.Atoms
            ase.Atoms instance from which the information is
            taken to create the Posinp.
        """
        return cls.from_dict(ase_atoms_to_pos_dict(atoms))

    @property
    def atoms(self):
        r"""
        Returns
        -------
        list of :class:`Atom`
            Atoms of the system (atomic type and positions).
        """
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if isinstance(atoms, list):
            if all([isinstance(at, Atom) for at in atoms]):
                self._atoms = atoms
            else:
                raise TypeError("All atoms should be mlcalcdriver.base.Atom instances")
        else:
            raise TypeError("Atoms should be given in a list")

    @property
    def units(self):
        r"""
        Returns
        -------
        str
            Units used to represent the atomic positions.
        """
        return self._units

    @units.setter
    def units(self, units):
        if isinstance(units, str):
            units = units.lower()
            if units.endswith("d0"):
                units = units[:-2]
            if units in ["angstrom", "atomic", "reduced"]:
                self._units = units
            else:
                raise ValueError("Units are not recognized.")
        else:
            raise TypeError("Units should be given as a string.")

    @property
    def boundary_conditions(self):
        r"""
        Returns
        -------
        str
            Boundary conditions.
        """
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        boundary_conditions = boundary_conditions.lower()
        if boundary_conditions in ["free", "periodic", "surface"]:
            self._boundary_conditions = boundary_conditions
        else:
            raise ValueError(
                "Boundary conditions {} are not recognized.".format(boundary_conditions)
            )

    @property
    def cell(self):
        r"""
        Returns
        -------
        ase.cell.Cell
            Object containing informations on the simulation cell.
            Zeros are used in the non-periodic directions.
        """
        return self._cell

    @cell.setter
    def cell(self, cell):
        if len(cell) == 2:
            cell, angles = cell
        else:
            angles = None
        if isinstance(cell, Cell):
            self._cell = cell
        elif cell is None:
            self._cell = Cell.new()
        elif isinstance(cell, list) or isinstance(cell, np.ndarray):
            if isinstance(cell, list):
                cell = np.array(
                    [
                        abs(float(size)) if size not in [".inf", "inf"] else 0.0
                        for size in cell
                    ]
                )
            if cell.size == 3 and angles is not None:
                if len(angles) == 3:
                    cell = np.concatenate([cell, angles])
                else:
                    raise ValueError("Need three angles to define a cell.")
            if cell.size in [3, 6, 9]:
                self._cell = Cell.new(cell)
            else:
                raise ValueError(
                    "Cell definition is not valid. See ase.cell.Cell documentation."
                )
        else:
            raise ValueError(
                "Cell definition is not valid. See ase.cell.Cell documentation."
            )

    @property
    def angles(self):
        r"""
        Returns
        -------
        numpy.array of three `float` or `None`
            Angles (degrees) between lattice vectors in order (yz, xz, xy)
        """
        return self.cell.angles()

    @property
    def positions(self):
        r"""
        Returns
        -------
        2D :class:`numpy.array` of shape (:math:`n_{at}`, 3)
            Position of all the atoms in the system.
        """
        return np.array([atom.position for atom in self])

    @property
    def masses(self):
        r"""
        Returns
        -------
        :class:`numpy.array` of length :math:`n_{at}`
            Masses of all the atoms in the system.
        """
        return np.array([atom.mass for atom in self])

    def __getitem__(self, index):
        r"""
        The items of a Posinp instance actually are the atoms (so as to
        behave like an immutable list of atoms).

        Parameters
        ----------
        index : int
            Index of a given atom

        Returns
        -------
        Atom
            The required atom.
        """
        return self.atoms[index]

    def __len__(self):
        return len(self.atoms)

    def __eq__(self, other):
        r"""
        Parameters
        ----------
        other : object
            Any other object.

        Returns
        -------
        bool
            `True` if both initial positions have the same number of
            atoms, the same units and boundary conditions and the same
            atoms (whatever the order of the atoms in the initial list
            of atoms).
        """
        try:
            # Check that both cells are the same, but first make sure
            # that the unimportant cell size are not compared
            cell = deepcopy(self.cell)
            other_cell = deepcopy(other.cell)
            same_cell = (cell == other_cell).all()
            # Check the other basic attributes
            same_BC = self.boundary_conditions == other.boundary_conditions
            same_base = (
                same_BC
                and len(self) == len(other)
                and self.units == other.units
                and same_cell
            )
            # Finally check the atoms only if the base is similar, as it
            # might be time-consuming for large systems
            if same_base:
                same_atoms = all([atom in other.atoms for atom in self.atoms])
                return same_atoms
            else:
                return False
        except AttributeError:
            return False

    def __str__(self):
        r"""
        Convert the Posinp to a string in the xyz format.

        Returns
        -------
        str
            The Posinp instance as a string.
        """
        # Create the first two lines of the posinp file
        pos_str = "{}   {}\n".format(len(self), self.units)
        pos_str += self.boundary_conditions
        if self.cell is not None:
            pos_str += "   {}   {}   {}\n".format(*self.cell.lengths())
        else:
            pos_str += "\n"
        if not self.orthorhombic:
            pos_str += "angles {}  {}  {}\n".format(*self.cell.angles())
        # Add all the other lines, representing the atoms
        pos_str += "".join([str(atom) for atom in self])
        return pos_str

    def __repr__(self):
        r"""
        Returns
        -------
            The string representation of a Posinp instance.
        """
        return (
            "Posinp({0.atoms}, '{0.units}', '{0.boundary_conditions}', "
            "cell={0.cell}, angles={0.angles})".format(self)
        )

    def write(self, filename):
        r"""
        Write the Posinp on disk.

        Parameters
        ----------
        filename : str
            Name of the input positions file.
        """
        with open(filename, "w") as stream:
            stream.write(str(self))

    def distance(self, i_at_1, i_at_2):
        r"""
        Evaluate the distance between two atoms.

        Parameters
        ----------
        i_at_1: int
            Index of the first atom.
        i_at_2: int
            Index of the second atom.

        Returns
        -------
        float
            Distance between both atoms.


        >>> atoms = [Atom('N', [0, 0, 0]), Atom('N', [3, 4, 0])]
        >>> pos = Posinp(atoms, units="angstrom", boundary_conditions="free")
        >>> assert pos.distance(0, 1) == 5.0
        """
        pos_1 = self[i_at_1].position
        pos_2 = self[i_at_2].position
        return np.linalg.norm(pos_1 - pos_2)

    def translate_atom(self, i_at, vector):
        r"""
        Translate the `i_at` atom along the three space coordinates
        according to the value of `vector`.

        Parameters
        ----------
        i_at : int
            Index of the atom.
        vector : list or :class:`numpy.array` of length 3
            Translation vector to apply.

        Returns
        -------
        Posinp
            New posinp where the `i_at` atom was translated by `vector`.


        .. Warning::

            You have to make sure that the units of the vector match
            those used by the posinp.


        >>> posinp = Posinp([Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])],
        ...                 'angstrom', 'free')
        >>> new_posinp = posinp.translate_atom(1, [0.0, 0.0, 0.05])
        >>> print(new_posinp)
        2   angstrom
        free
        N   0.0   0.0   0.0
        N   0.0   0.0   1.15
        <BLANKLINE>
        """
        new_posinp = deepcopy(self)
        new_posinp.atoms[i_at] = self[i_at].translate(vector)
        return new_posinp

    def translate(self, vector):
        r"""
        Translate all the atoms along the three space coordinates
        according to the value of `vector`.

        Parameters
        ----------
        vector : list or :class:`numpy.array` of length 3
            Translation vector to apply.

        Returns
        -------
        Posinp
            New posinp where all the atoms were translated by `vector`.


        .. Warning::

            You have to make sure that the units of the vector match
            those used by the posinp.
        """
        new_positions = self.positions + np.array(vector)
        atoms = [
            Atom(atom.type, pos.tolist())
            for atom, pos in zip(self.atoms, new_positions)
        ]
        return Posinp(
            atoms,
            units=self.units,
            cell=self.cell,
            boundary_conditions=self.boundary_conditions,
        )

    def to_centroid(self):
        r"""
        Center the system to its centroid (*i.e.*, geometric center).

        Returns
        -------
        Posinp
            New posinp where all the atoms are centered on the geometric
            center of the system.
        """
        centroid = np.average(self.positions, axis=0)
        return self.translate(-centroid)

    def to_barycenter(self):
        r"""
        Center the system to its barycenter (*i.e.*, center of mass).

        Returns
        -------
        Posinp
            New posinp where all the atoms are centered on the center
            of mass of the system.
        """
        m = self.masses
        barycenter = np.sum(m * self.positions.T, axis=1) / np.sum(m)
        return self.translate(-barycenter)

    def convert_units(self, new_units):
        r"""
        Converts the atomic positions in another units.

        Parameters
        ----------
        new_units: str
            The new units in which the positions should be converted.
            Can either be "angstrom" or "atomic".
        """
        if new_units not in ["angstrom", "atomic"]:
            raise ValueError("New units are not recognized.")
        if self.units == new_units:
            pass
        elif self.units == "atomic" and new_units == "angstrom":
            for atom in self:
                atom.position = atom.position * B_TO_ANG
            self.cell = Cell.new(self.cell * B_TO_ANG)
        elif self.units == "angstrom" and new_units == "atomic":
            for atom in self:
                atom.position = atom.position * ANG_TO_B
            self.cell = Cell.new(self.cell * ANG_TO_B)
        elif self.units == "reduced" and new_units == "atomic":
            for atom in self:
                atom.position = np.sum(atom.position * self.cell, axis=0)
        elif self.units == "reduced" and new_units == "angstrom":
            for atom in self:
                atom.position = np.sum(atom.position * self.cell * B_TO_ANG, axis=0)
            self.cell = Cell.new(self.cell * B_TO_ANG)
        else:
            raise NotImplementedError
        self.units = new_units

    def angle(self, i, j, k):
        r"""
        Returns the angle between three atoms

        Parameters
        ----------
        i: int
            Index of the first atom
        j: int
            Index of the middle atom
        k: int
            Index of the third atom

        Returns
        -------
        angle: float
            Angle between the three atoms, in radians
        """
        ij = self.positions[i] - self.positions[j]
        jk = self.positions[k] - self.positions[j]
        angle = np.arccos(
            np.clip(
                np.dot(ij, jk) / (np.linalg.norm(ij) * np.linalg.norm(jk)), -1.0, 1.0
            )
        )
        return angle


class Atom(object):
    r"""
    Class allowing to represent an atom by its type and position.
    """

    def __init__(self, atom_type, position, isotope=None):
        r"""
        Parameters
        ----------
        atom_type : str
            Type of the atom.
        position : list or :class:`numpy.array` of length 3
            Position of the atom.


        >>> a = Atom('C', [0, 0, 0])
        >>> a.type
        'C'
        >>> a.position
        array([0., 0., 0.])
        >>> a.mass
        12.011
        """
        # TODO: Check that the atom type exists
        self.type = atom_type
        self.position = position
        self.isotope = isotope
        # self.mass = ATOMS_MASS[self.isotope] if self.isotope else ATOMS_MASS[self.type]

    @classmethod
    def from_dict(cls, atom_dict):
        r"""
        Parameters
        ----------
        atom_dict : dict
            Information about an atom given by a dict whose only key is
            the atom type and the value is the atomic position. This
            format is mainly found in bigdft logfiles.
        """
        [(atom_type, position)] = atom_dict.items()
        return cls(atom_type, position)

    @property
    def type(self):
        r"""
        Returns
        -------
        str
            Type of the atom.
        """
        return self._type

    @type.setter
    def type(self, type):
        if isinstance(type, str):
            self._type = type
        else:
            TypeError("Atom type should be given as a string.")

    @property
    def position(self):
        r"""
        Returns
        -------
        list or :class:`numpy.array` of length 3
            Position of the atom in cartesian coordinates.
        """
        return self._position

    @position.setter
    def position(self, position):
        assert len(position) == 3, "The position must have three components."
        self._position = np.array(position, dtype=float)

    @property
    def isotope(self):
        r"""
        Returns
        -------
        str or None
            Isotope of the atom
        """
        return self._isotope

    @isotope.setter
    def isotope(self, isotope):
        self._isotope = isotope

    @property
    def mass(self):
        r"""
        Returns
        -------
        float
            Mass of the atom in atomic mass units.
        """
        return ATOMS_MASS[self.isotope] if self.isotope else ATOMS_MASS[self.type]

    def translate(self, vector):
        r"""
        Translate the coordinates of the atom by the values of the
        vector.

        Returns
        -------
        Atom
            Atom translated according to the given vector.

        Parameters
        ----------
        vector : list or :class:`numpy.array` of length 3
            Translation vector to apply.


        >>> Atom('C', [0, 0, 0]).translate([0.5, 0.5, 0.5])
        Atom('C', [0.5, 0.5, 0.5])
        """
        assert len(vector) == 3, "The vector must have three components"
        new_atom = deepcopy(self)
        new_atom.position = self.position + np.array(vector)
        return new_atom

    def __str__(self):
        r"""
        Returns
        -------
        str
            String representation of the atom, mainly used to create the
            string representation of a Posinp instance.
        """
        return (
            f"{self.type}  {self.position[0]:.15}  {self.position[1]:.15}  {self.position[2]:.15}  {self.isotope}\n"
            if self.isotope
            else f"{self.type}  {self.position[0]:.15}  {self.position[1]:.15}  {self.position[2]:.15}\n"
        )

    def __repr__(self):
        r"""
        Returns
        -------
        str
            General string representation of an Atom instance.
        """
        return (
            "Atom('{}', {}, {})".format(self.type, list(self.position), self.isotope)
            if self.isotope
            else "Atom('{}', {})".format(self.type, list(self.position))
        )

    def __eq__(self, other):
        r"""
        Two atoms are the same if they are located on the same position
        and have the same type.

        Parameters
        ----------
        other
            Other object.

        Returns
        -------
        bool
            True if both atoms have the same type and position.


        >>> a = Atom('C', [0., 0., 0.])
        >>> a == 1
        False
        >>> a == Atom('N', [0., 0., 0.])
        False
        >>> a == Atom('C', [1., 0., 0.])
        False
        """
        try:
            return (
                np.allclose(self.position, other.position) and self.type == other.type
            )
        except AttributeError:
            return False
