r"""
The :class:`Job` class is the base object defining similar machine learning
predictions for a single or many atomic configurations.
"""

import numpy as np
import torch
from copy import deepcopy
from mlcalcdriver.calculators import Calculator
from mlcalcdriver.base import Posinp
import warnings


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
        posinp : :class:`Posinp` or list of :class:`Posinp`
            Atomic positions for the job. Many different configurations
            may be predicted at the same time, in that case they should
            be passed in a list.
        calculator : :class:`Calculator`
            `Calculator` instance to use to evaluate
            properties in the run() method.
        """
        self.name = name
        self.posinp = posinp
        self.num_struct = len(self.posinp)
        self.calculator = calculator
        self.results = JobResults(
            positions=self.posinp,
            properties=self.calculator.available_properties,
        )

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
                    """
                    Atomic positions should be given
                    only in Posinp instances.
                    """
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
        r"""
        Returns
        -------
        JobResults
            The dictionnary containing the results of the calculation
        """
        return self._results

    @results.setter
    def results(self, results):
        self._results = results

    @property
    def calculator(self):
        r"""
        Returns
        -------
        Calculator
            The Calculator object to use for the Job
        """
        return self._calculator

    @calculator.setter
    def calculator(self, calculator):
        if isinstance(calculator, Calculator):
            self._calculator = calculator
        else:
            raise TypeError(
                """
                The calculator for the Job must be a class or a
                metaclass derived from mlcalcdriver.calculators.Calculator.
                """
            )

    def run(self, property, device="cpu", batch_size=128):
        r"""
        Main method to call to obtain results for a Job

        Parameters
        ----------
        property : str
            Property to calculate. Must be in the
            available_properties of the Calculator except the
            forces which can be derived from an energy
            Calculator.
        device : str
            Device on which to run the calculation.
            Either `"cpu"` or `"cuda"` to run on cpu or gpu.
            Default is `"cpu"` and should not be changed, except
            for very large systems.
        batch_size : int
            Size of the mini-batches used in predictions.
            Default is 128.
        """
        device = str(device)
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                warnings.warn(
                    "CUDA was asked for, but is not available.", UserWarning
                )

        if property not in self.calculator.available_properties:
            if not (
                property == "forces"
                and "energy" in self.calculator.available_properties
            ):
                raise ValueError(
                    "The property {} is not available".format(property)
                )
            else:
                self._create_additional_structures()
                raw_predictions = self.calculator.run(
                    property="energy", posinp=self.posinp
                )
                pred_idx = 0
                predictions = {}
                predictions["energy"], predictions["forces"] = [], []
                for struct_idx in range(self.num_struct):
                    predictions["energy"].append(
                        raw_predictions["energy"][pred_idx][0]
                    )
                    pred_idx += 1
                    predictions["forces"].append(
                        self._calculate_forces(
                            raw_predictions["energy"][
                                pred_idx : pred_idx
                                + 12 * len(self._init_posinp[struct_idx])
                            ]
                        )
                    )
                    pred_idx += 12 * len(self._init_posinp[struct_idx])
                self.posinp = deepcopy(self._init_posinp)
        else:
            predictions = self.calculator.run(
                property=property, posinp=self.posinp
            )
        for pred in predictions.keys():
            self.results.update({pred: predictions[pred]})

    def _create_additional_structures(self, deriv_length=0.015):
        r"""
        Creates the additional structures needed to do a numeric
        derivation of the energy to calculate the forces.
        """
        self._init_posinp = deepcopy(self.posinp)
        self._deriv_length = deriv_length
        all_structs = []
        # Second order forces calculations
        for str_idx, struct in enumerate(self.posinp):
            all_structs.append(struct)
            for factor in [2, 1, -1, -2]:
                for dim in [
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                ]:
                    all_structs.extend(
                        [
                            struct.translate_atom(
                                atom_idx, deriv_length * factor * dim
                            )
                            for atom_idx in range(len(struct))
                        ]
                    )
        self.posinp = all_structs

    def _calculate_forces(self, predictions):
        r"""
        Method to calculate forces from the displaced atomic positions

        Parameters
        ----------
        predictions : 1D numpy array (size 6*n_at)
             Contains the predictions obtained from the neural network

        Returns
        -------
        forces : 2D numpy array (size (n_at, 3))
            Forces for each structure
        """
        nat = int(len(predictions) / 12)
        forces = np.zeros((nat, 3))
        for i in range(3):
            ener1, ener2, ener3, ener4 = (
                predictions[np.arange(i * nat, (i + 1) * nat, 1)],
                predictions[np.arange((i + 3) * nat, (i + 4) * nat, 1)],
                predictions[np.arange((i + 6) * nat, (i + 7) * nat, 1)],
                predictions[np.arange((i + 9) * nat, (i + 10) * nat, 1)],
            )
            forces[:, i] = -(
                (-ener1 + 8 * (ener2 - ener3) + ener4).reshape(nat)
                / (12 * self._deriv_length)
            )
        return forces


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
        positions : :class:`Posinp` or list of :class:`Posinp`
            Atomic positions used in the Job
        properties : str or list of str
            Property or properties that are returned by the chosen
            model.
        """
        self.positions = positions
        self.n_at = [len(pos) for pos in self.positions]
        self.atom_types = [
            set([atom.type for atom in pos]) for pos in self.positions
        ]
        self.boundary_conditions = [
            pos.boundary_conditions for pos in self.positions
        ]
        self.cell = [pos.cell for pos in self.positions]

        self.properties = properties
        for prop in self.properties:
            self[prop] = None

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
            raise (
                "Properties should be given as a string or a list of strings."
            )

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
