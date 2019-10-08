r"""
The :class:`Job` class is the base object defining similar machine learning
predictions for a single or many atomic configurations.
"""

import numpy as np
from copy import deepcopy
from mlcalcdriver.base import Posinp


class Job:
    r"""
    This class defines a machine learning prediction. It must
    contain a mlcalcdriver.Posinp instance to define the atomic
    configuration.
    """

    def __init__(self, name="", posinp=None, calculator=None):
        r"""
        Parameters
        ----------
        name : str
            Name of the job. Will be used to name the created files.
        posinp : Posinp or list of Posinp
            Atomic positions for the job. Many different configurations
            may be predicted at the same time, in that case they should
            be passed in a list.
        """
        self.name = name
        self.posinp = posinp
        self.num_struct = len(posinp)
        self.results = JobResults(positions=self.posinp, properties="energy")

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp
            Initial positions of the prediction
        """
        return self._posinp

    @posinp.setter
    def posinp(self, posinp):
        if posinp is None:
            raise ValueError("A Job instance has no initial positions.")
        elif not isinstance(posinp, list):
            posinp = [posinp]
        for pos in posinp:
            if not isinstance(pos, Posinp):
                raise TypeError(
                    "Atomic positions should be given only in Posinp instances."
                )
        self._posinp = posinp

    @property
    def name(self):
        r"""
        Returns
        -------
        str
            Base name of the prediction used to set the names
            of files and directories.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def num_struct(self):
        r"""
        Returns
        -------
        int
            Number of different structures when Job is declared
        """
        return self._num_struct

    @num_struct.setter
    def num_struct(self, num_struct):
        self._num_struct = num_struct

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, results):
        self._results = results


class JobResults(dict):
    r"""
    Dictionnary containing results from a Job after the run() method
    is completed. A JobResults instance is created for each Job, and
    the results of the latter should be accessed through the former,
    by using the `Job.results` property.

    Predicted values can be accessed as in a standard dictionnary


    >>> energy = Job.results["energy"]
    >>> type(energy)
    <class 'list'>

    The returned values will be `None` if the Job was not complete.
    Otherwise, the list contains one value for each structure in the Job.
    """

    def __init__(self, positions, properties):
        r"""
        Parameters
        ----------
        properties : str or list of str
            Property or properties that are returned by the chosen
            model. 
        """
        self.positions = positions
        self.n_at = [len(pos) for pos in self.positions]
        self.atom_types = [set([atom.type for atom in pos]) for pos in self.positions]
        self.boundary_conditions = [pos.boundary_conditions for pos in self.positions]
        self.cell = [pos.cell for pos in self.positions]

        self.properties = properties
        for prop in self.properties:
            self[prop] = None

    def _store_results(self):
        r"""
        Method to store results from a calculation in
        the JobResults object.
        """

    @property
    def properties(self):
        return self["properties"]

    @properties.setter
    def properties(self, properties):
        if isinstance(properties, str):
            properties = [properties]
        if isinstance(properties, list):
            if any([not isinstance(prop, str) for prop in properties]):
                raise ("All properties should be given as a string.")
            else:
                self["properties"] = properties
        else:
            raise ("Properties should be given as a string or a list of strings.")

    @property
    def positions(self):
        r"""
        Returns
        -------
        list of Posinps
            List containing the base Posinp objects for the predictions.
        """
        return self["positions"]

    @positions.setter
    def positions(self, positions):
        self["positions"] = positions

    @property
    def n_at(self):
        r"""
        Returns
        -------
        list of ints
            List containing the number of atoms of each structure.
        """
        return self["n_at"]

    @n_at.setter
    def n_at(self, n_at):
        self["n_at"] = n_at

    @property
    def atom_types(self):
        r"""
        Returns
        -------
        list of sets
            List containing sets of the elements present in each structure.
        """
        return self["atom_types"]

    @atom_types.setter
    def atom_types(self, atom_types):
        self["atom_types"] = atom_types

    @property
    def boundary_conditions(self):
        r"""
        Returns
        -------
        list of strings
            List containing the boundary conditions, either `free`,
            `surface` or `periodic`, of each structure.
        """
        return self["boundary_conditions"]

    @boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        self["boundary_conditions"] = boundary_conditions

    @property
    def cell(self):
        r"""
        Returns
        -------
        list of lists of floats
            List containing cell dimensions of each structure,
            None for free boundary conditions.
        """
        return self["cell"]

    @cell.setter
    def cell(self, cell):
        self["cell"] = cell
