r"""
Calculator metaclass to accomodate machine learning models
trained using the SchnetPack package.
"""

import os
import numpy as np
import torch
from schnetpack import AtomsLoader
from mlcalcdriver.globals import eVA
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.calculators.utils import torch_derivative, get_derivative_names
from mlcalcdriver.interfaces import posinp_to_ase_atoms, SchnetPackData
from schnetpack.environment import SimpleEnvironmentProvider, AseEnvironmentProvider
from schnetpack.utils import load_model
from mlcalcdriver.globals import EV_TO_HA, B_TO_ANG

#from functorch import grad


class LowMemSPCalculator(SchnetPackCalculator):
    r"""
    Calculator based on a SchnetPack model
    """

    def __init__(
        self,
        model_dir,
        available_properties=None,
        device="cpu",
        units=eVA,
    ):
        super(LowMemSPCalculator, self).__init__(model_dir=model_dir, available_properties=["energy"], device=device, units=units)
        for module in self.model.output_modules:
            module.derivative = None

    def run(
        self, property, posinp=None, batch_size=128,
    ):
        r"""
        Central method to use when making a calculation with
        the calculator.

        Parameters
        ----------
        property : str
            Property to be predicted by the calculator
        posinp : Posinp
            Atomic configuration to pass to the model
        batch_size : int
            Batch sizes. Default is 128.

        Returns
        -------
        predictions : :class:`numpy.ndarray`
            Corresponding prediction by the model.
        """
        init_property, out_name, derivative, wrt = get_derivative_names(
            property, self.available_properties
        )

        if len(posinp) > 1:
            raise RuntimeError("The low memory version of the calculator can only be used on one structure at a time.")

        data = [posinp_to_ase_atoms(pos) for pos in posinp]
        pbc = True if any(pos.pbc.any() for pos in data) else False
        environment_provider = (
            AseEnvironmentProvider(cutoff=self.cutoff)
            if pbc
            else SimpleEnvironmentProvider()
        )

        data = SchnetPackData(
            data=data,
            environment_provider=environment_provider,
            collect_triples=self.model_type == "wacsf",
        )
        data_loader = AtomsLoader(data, batch_size=1)

        n_atoms = len(posinp[0])
        if derivative == 0:
            results = np.zeros(n_atoms)
            with torch.no_grad():
                for batch in data_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    for i in range(n_atoms):
                        results[i] = self.model(batch, at_idx=i)[property].cpu().detach().numpy()
            prediction = {property: np.sum(results)}
        if abs(derivative) == 1:
            #def forward_pass(model, batch, positions):
            #    batch["_positions"] = positions
            #    r = model(batch)["energy"]
            #    print(r)
            #    return torch.squeeze(r)
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch[wrt[0]].requires_grad_()
                #positions = batch.pop("_positions")
                #deriv1 = grad(lambda positions: forward_pass(self.model, batch, positions))(positions)

                results = self.model(batch)
                deriv1 = torch.unsqueeze(
                    torch_derivative(results[init_property], batch[wrt[0]]), 0
                )
                if derivative < 0:
                    deriv1 = -1.0 * deriv1
                pred.append({out_name: deriv1})
        if abs(derivative) == 2:
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                for inp in set(wrt):
                    batch[inp].requires_grad_()
                results = self.model(batch)
                deriv2 = torch.unsqueeze(
                    torch_derivative(
                        torch_derivative(
                            results[init_property], batch[wrt[0]], create_graph=True,
                        ),
                        batch[wrt[0]],
                    ),
                    0,
                )
                if derivative < 0:
                    deriv2 = -1.0 * deriv2
                pred.append({out_name: deriv2})
        #predictions = {}
        #if derivative:
        #    predictions[property] = np.concatenate(
        #        [batch[out_name].cpu().detach().numpy() for batch in pred]
        #    )
        #else:
        #    predictions[property] = np.concatenate(
        #        [batch[init_property].cpu().detach().numpy() for batch in pred]
        #    )
        return prediction
