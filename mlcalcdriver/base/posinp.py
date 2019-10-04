r"""
The :class:`Posinp` and :class:`Atom` classes are represent the
atomic systems used as input for a calculation.
"""

from copy import deepcopy
from collections.abc import Sequence
import numpy as np
from mlcalcdriver.globals import ATOMS_MASS


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

    def __init__(self, atoms, units="angstroem", boundary_conditions="free", cell=None):
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
        ...                 'angstroem', 'free')
        >>> len(posinp)
        2
        >>> posinp.boundary_conditions
        'free'
        >>> posinp.units
        'angstroem'
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
        self.cell = cell
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
        if cell is not None:
            if len(cell) != 3:
                raise ValueError(
                    "The cell size must be of length 3 (one value per "
                    "space coordinate)"
                )
        else:
            if boundary_conditions != "free":
                raise ValueError(
                    "You must give a cell size to use '{}' boundary conditions".format(
                        boundary_conditions
                    )
                )
        if boundary_conditions == "periodic" and "inf" in cell:
            raise ValueError(
                "Cannot use periodic boundary conditions with a cell meant "
                "for a surface calculation."
            )
        elif boundary_conditions == "free" and units == "reduced":
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
        if boundary_conditions != "free":
            cell = line2[1:4]
        else:
            cell = None
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
            position = line[1:4]
            atoms.append(Atom(atom_type, position))
        return cls(atoms, units, boundary_conditions, cell=cell)

    @property
    def atoms(self):
        r"""
        Returns
        -------
        list of Atoms
            Atoms of the system (atomic type and positions).
        """
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if isinstance(atoms, list):
            if all([isinstance(at, Atom) for at in atoms]):
                self._atoms = atoms
            else:
                raise TypeError("All atoms should be mybigdft.Atoms instances")
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
            self._units = units
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
        list of three float or None
            Cell size.
        """
        return self._cell

    @cell.setter
    def cell(self, cell):
        if cell is not None:
            cell = [
                abs(float(size)) if size not in [".inf", "inf"] else "inf"
                for size in cell
            ]
        self._cell = cell

    @property
    def positions(self):
        r"""
        Returns
        -------
        2D numpy array of shape (:math:`n_{at}`, 3)
            Position of all the atoms in the system.
        """
        return np.array([atom.position for atom in self])

    @property
    def masses(self):
        r"""
        Returns
        -------
        numpy array of length :math:`n_{at}`
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
            if self.boundary_conditions == "surface":
                cell[1] = 0.0
            if other.boundary_conditions == "surface":
                other_cell[1] = 0.0
            same_cell = cell == other_cell
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
            pos_str += "   {}   {}   {}\n".format(*self.cell)
        else:
            pos_str += "\n"
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
            "cell={0.cell})".format(self)
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
        >>> pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        >>> assert pos.distance(0, 1) == 5.0
        """
        pos_1 = self[i_at_1].position
        pos_2 = self[i_at_2].position
        return np.sqrt(sum([(pos_1[i] - pos_2[i]) ** 2 for i in range(3)]))

    def translate_atom(self, i_at, vector):
        r"""
        Translate the `i_at` atom along the three space coordinates
        according to the value of `vector`.

        Parameters
        ----------
        i_at : int
            Index of the atom.
        vector : list or numpy.array of length 3
            Translation vector to apply.

        Returns
        -------
        Posinp
            New posinp where the `i_at` atom was translated by `vector`.


        .. Warning::

            You have to make sure that the units of the vector match
            those used by the posinp.


        >>> posinp = Posinp([Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])],
        ...                 'angstroem', 'free')
        >>> new_posinp = posinp.translate_atom(1, [0.0, 0.0, 0.05])
        >>> print(new_posinp)
        2   angstroem
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
        vector : list or numpy.array of length 3
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


class Atom(object):
    r"""
    Class allowing to represent an atom by its type and position.
    """

    def __init__(self, atom_type, position):
        r"""
        Parameters
        ----------
        atom_type : str
            Type of the atom.
        position : list or numpy.array of length 3
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
        self.mass = ATOMS_MASS[self.type]

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
        list or numpy.array of length 3
            Position of the atom in cartesian coordinates.
        """
        return self._position

    @position.setter
    def position(self, position):
        assert len(position) == 3, "The position must have three components."
        self._position = np.array(position, dtype=float)

    @property
    def mass(self):
        r"""
        Returns
        -------
        float
            Mass of the atom in atomic mass units.
        """
        return self._mass

    @mass.setter
    def mass(self, mass):
        self._mass = mass

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
        vector : list or numpy.array of length 3
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
        return "{t}  {: .15}  {: .15}  {: .15}\n".format(t=self.type, *self.position)

    def __repr__(self):
        r"""
        Returns
        -------
        str
            General string representation of an Atom instance.
        """
        return "Atom('{}', {})".format(self.type, list(self.position))

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
