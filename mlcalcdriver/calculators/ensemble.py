import torch
import numpy as np
import mlcalcdriver.base as base
from mlcalcdriver.globals import eVA
import mlcalcdriver.calculators as mlc


class Ensemble:
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
        return {property: result}


class EnsembleCalculator(mlc.Calculator):
    def __init__(self, modelpaths, device="cpu", available_properties=None, units=eVA):
        self.ensemble = Ensemble(modelpaths, device=device, units=units)
        super(EnsembleCalculator, self).__init__(
            available_properties=available_properties, units=units
        )

    @property
    def ensemble(self):
        return self._ensemble

    @ensemble.setter
    def ensemble(self, ensemble):
        self._ensemble = ensemble

    def run(self, property, posinp=None):
        return self.ensemble.run(property, posinp=posinp)

    def _get_available_properties(self):
        all_props = [model.available_properties for model in self.ensemble.models]
        avail_prop = []
        for prop in all_props[0]:
            if all(prop in el for el in all_props):
                avail_prop.append(prop)
        return avail_prop
