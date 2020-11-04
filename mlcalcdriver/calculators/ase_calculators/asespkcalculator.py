from ase.calculators.calculator import Calculator, all_changes
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.base import Posinp, Job
from copy import deepcopy
import numpy as np


class AseSpkCalculator(Calculator):

    def __init__(self, model_dir, available_properties=None, device="cpu", **kwargs):
        Calculator.__init__(self, **kwargs)
        self.schnetpackcalculator = SchnetPackCalculator(
            model_dir=model_dir,
            available_properties=available_properties,
            device=device,
        )
        self.implemented_properties = self.schnetpackcalculator._get_available_properties()
        if "energy" in self.implemented_properties and "forces" not in self.implemented_properties:
            self.implemented_properties.append("forces")

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)
            posinp = Posinp.from_ase(atoms)

            job = Job(posinp=posinp, calculator=self.schnetpackcalculator)
            for prop in properties:
                job.run(prop)
            results = {}
            for prop, result in zip(job.results.keys(), job.results.values()):
                results[prop] = np.squeeze(result)
            self.results = results
