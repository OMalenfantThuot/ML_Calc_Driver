from ase.calculators.calculator import Calculator, all_changes
from mlcalcdriver.calculators import SchnetPackCalculator
from copy import deepcopy
import numpy as np


class AseSpkCalculator(Calculator):
    r"""
    Wrapper :class:`Calculator` class around the :class:`SchnetPackCalculator`
    class to use directly inside ASE funtions.
    """

    def __init__(
        self,
        model_dir,
        available_properties=None,
        device="cpu",
        md=False,
        dropout=False,
        **kwargs
    ):
        r"""
        Parameters
        ----------
        model_dir : str
            Same as :class:`SchnetPackCalculator`.
        available_properties : str or list of str
            Same as :class:`SchnetPackCalculator`.
        device : str
            Same as :class:`SchnetPackCalculator`.
        md : bool
            Default is False. Should be set to True if the
            calculator is used for molecular dynamics.
        dropout : bool
            Same as :class:`SchnetPackCalculator`.
        units : dict
            Same as :class:`SchnetPackCalculator`.
        """
        Calculator.__init__(self, **kwargs)
        self.schnetpackcalculator = SchnetPackCalculator(
            model_dir=model_dir,
            available_properties=available_properties,
            device=device,
            md=md,
            dropout=dropout,
        )
        self.implemented_properties = (
            self.schnetpackcalculator._get_available_properties()
        )
        if (
            "energy" in self.implemented_properties
            and "forces" not in self.implemented_properties
        ):
            self.implemented_properties.append("forces")

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        r"""
        This method will be called by ASE functions.
        """
        if self.calculation_required(atoms, properties):
            from mlcalcdriver.base.posinp import Posinp

            Calculator.calculate(self, atoms)
            posinp = Posinp.from_ase(atoms)

            from mlcalcdriver.base.job import Job

            job = Job(posinp=posinp, calculator=self.schnetpackcalculator)
            for prop in properties:
                job.run(prop)
            results = {}
            for prop, result in zip(job.results.keys(), job.results.values()):
                results[prop] = np.squeeze(result)
            self.results = results
