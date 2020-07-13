r"""
The :class:`Geopt` class allows to perform a geometry optimization to
relax the forces on a given structure, using a machine
learning model.
"""

import numpy as np
from copy import deepcopy
from mlcalcdriver import Posinp, Job
from mlcalcdriver.calculators import Calculator


class Geopt:
    r"""
    This class allows to relax the input geometry of a given system in
    order to find the structure that minimizes the forces. The final
    result obtained depends on the trained machine learning model used.
    """

    def __init__(
        self, posinp, calculator, forcemax=0.01, step_size=0.002, max_iter=500
    ):
        r"""
        Parameters
        ----------
        posinp : mybigdft.Posinp
            Starting configuration to relax
        calculator : Calculator
            mlcalcdriver.Calculator instance that will be used in
            the created Job to evaluate properties.
        forcemax : float
            Stopping criterion on the forces (in eV/Angstrom).
            Default is `0.01`.
        step_size : float
            Step size for each relaxation step. Default
            is `0.003` Angstrom<sup>2</sup>/eV.
        max_iter : int
            Maximum number of iterations. Default is 500.
        """
        self.posinp = posinp
        self.calculator = calculator
        self.forcemax = forcemax
        self.step_size = step_size
        self.max_iter = max_iter
        self.final_posinp = None

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp
            Initial posinp of the geometry optimization procedure
        """
        return self._posinp

    @posinp.setter
    def posinp(self, posinp):
        if posinp is None:
            raise ValueError("No initial positions were provided.")
        self._posinp = posinp

    @property
    def calculator(self):
        r"""
        Returns
        -------
        Calculator
            The Calculator object to use for the Jobs necessary to
            perform the geometry optimisation.
        """
        return self._calculator

    @calculator.setter
    def calculator(self, calculator):
        if isinstance(calculator, Calculator):
            self._calculator = calculator
        else:
            raise TypeError(
                """
                The calculator for the Geopt instance must be a class or a
                metaclass derived from mlcalcdriver.calculators.Calculator.
                """
            )

    @property
    def final_posinp(self):
        r"""
        Returns
        -------
        Posinp or None
            Final posinp of the geometry optimization or None if
            the the optimization has not been completed
        """
        return self._final_posinp

    @final_posinp.setter
    def final_posinp(self, final_posinp):
        self._final_posinp = final_posinp

    @property
    def forcemax(self):
        r"""
        Returns
        -------
        float
            Stopping criterion on the forces (in eV/Angstrom)
        """
        return self._forcemax

    @forcemax.setter
    def forcemax(self, forcemax):
        self._forcemax = forcemax

    @property
    def step_size(self):
        r"""
        Returns
        -------
        float
            Step size for each relaxation step
        """
        return self._step_size

    @step_size.setter
    def step_size(self, step_size):
        self._step_size = step_size

    @property
    def max_iter(self):
        r"""
        Returns
        -------
        int
            Maximum number of iterations
        """
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self._max_iter = int(max_iter)

    def run(self, batch_size=128, recenter=False, verbose=0):
        r"""
        Parameters
        ----------
        batch_size : int
            Size of the mini-batches used in predictions. Default is 128.
        recenter : bool
            If `True`, the structure is recentered on its
            centroid after the relaxation. Default is `False`.
        verbose : int
            Controls the verbosity of the output. If 0 (Default), no written output.
            If 1, a message will indicate if the optimization was succesful or not
            and the remaining forces. If 2 or more, each iteration will provide
            an output.
        """

        temp_posinp = deepcopy(self.posinp)
        verbose = int(verbose)

        # Optimization loop
        for i in range(1, self.max_iter + 1):
            # Forces calculation
            job = Job(posinp=temp_posinp, calculator=self.calculator)
            job.run("forces", batch_size=batch_size)
            # Moving the atoms
            for j in range(len(job.posinp[0])):
                temp_posinp = temp_posinp.translate_atom(
                    j, self.step_size * job.results["forces"][0][j]
                )
            fmax = np.max(np.abs(job.results["forces"][0]))
            if verbose >= 2:
                print(
                    "At iteration {}, the maximum remaining force is {:6.4f} eV/Ha.".format(
                        i, fmax
                    )
                )
            # Stopping condition
            if fmax < self.forcemax:
                if verbose >= 1:
                    print("Geometry optimization stopped at iteration {}.".format(i))
                break
            # Step size reduction to help forces optimization
            if i % 100 == 0:
                self.step_size = self.step_size * 0.9
            # Maximum iterations check
            if i == self.max_iter:
                if verbose >= 1:
                    print(
                        "Geometry optimization was not succesful at iteration {}.".format(
                            i
                        )
                    )
        if verbose >= 1:
            print("Max remaining force is {:6.4f}.".format(fmax))
        self.final_posinp = temp_posinp
        if recenter:
            self.final_posinp = self.final_posinp.to_centroid()
