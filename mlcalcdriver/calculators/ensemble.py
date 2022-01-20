import torch
import numpy as np
import mlcalcdriver.base as base
from mlcalcdriver.globals import eVA
import mlcalcdriver.calculators as mlc
from ase.calculators.calculator import Calculator, all_changes


class Ensemble:
    r"""
    Not a Calculator. This holds the models needed in the actual calculators.
    Only implemented for SchnetPack models, at the moment. Could be easily expanded.
    """
    def __init__(self, modelpaths, device="cpu", units=eVA):
        self.modelpaths = modelpaths
        self.models = self._load_models(device, units)

    @property
    def modelpaths(self):
        return self._modelpaths

    @modelpaths.setter
    def modelpaths(self, modelpaths):
        if not isinstance(modelpaths, (list, tuple, set)):
            raise TypeError("The modelpaths should be given in a list, tuple, or set.")
        self._modelpaths = modelpaths

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models):
        self._models = models

    def _load_models(self, device, units):
        models = []
        for path in self.modelpaths:
            try:
                models.append(
                    mlc.SchnetPackCalculator(path, device=device, units=units)
                )
            except Exception:
                raise Exception
        return models

    def run(self, property, posinp=None):
        results = []
        for i, model in enumerate(self.models):
            job = base.Job(posinp=posinp, calculator=model)
            job.run(property, batch_size=1)
            results.append(job.results[property][np.newaxis, ...])

        result = np.mean(np.concatenate(results, axis=0), axis=0)
        result_std = np.std(np.concatenate(results, axis=0), axis=0)
        return {property: result, property + "_std": result_std}


class EnsembleCalculator(mlc.Calculator):
    r"""
    Calculator using many similarly trained models to approximate
    a convfidence interval on predictions. Can be used with any :class:`Ensemble`.
    """
    def __init__(self, modelpaths, device="cpu", available_properties=None, units=eVA):
        self.ensemble = Ensemble(modelpaths, device=device, units=units)
        r"""
        Parameters
        ----------
        modelpaths : list, tuple or set of str
            Paths to the models
        The other parameters are the same as the base SchnetPackCalculators
        """
        super(EnsembleCalculator, self).__init__(
            available_properties=available_properties, units=units
        )

    @property
    def ensemble(self):
        return self._ensemble

    @ensemble.setter
    def ensemble(self, ensemble):
        self._ensemble = ensemble

    def run(self, property, posinp=None, batch_size=None):
        return self.ensemble.run(property, posinp=posinp)

    def _get_available_properties(self):
        all_props = [model.available_properties for model in self.ensemble.models]
        avail_prop = []
        for prop in all_props[0]:
            if all(prop in el for el in all_props):
                avail_prop.append(prop)
        return avail_prop


class AseEnsembleCalculator(Calculator):
    r"""
    Same thing as :class:`EnsembleCalculator`, but interfaced to use in ASE.
    """
    def __init__(self, modelpaths, available_properties=None, device="cpu", **kwargs):
        Calculator.__init__(self, **kwargs)
        self.ensemblecalc = EnsembleCalculator(
            modelpaths=modelpaths,
            device=device,
            available_properties=available_properties,
        )
        self.implemented_properties = (
            self.ensemblecalc._get_available_properties()
        )
        if (
            "energy" in self.implemented_properties
            and "forces" not in self.implemented_properties
        ):
            self.implemented_properties.append("forces")

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)
            posinp = base.Posinp.from_ase(atoms)

        job = base.Job(posinp=posinp, calculator=self.ensemblecalc)
        for prop in properties:
            job.run(prop)
        results = {}
        for prop, result in zip(job.results.keys(), job.results.values()):
            results[prop] = np.squeeze(result)
        self.results = results
