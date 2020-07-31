r"""
Calculator metaclass to accomodate machine learning models
trained using the SchnetPack package.
"""

import os
import numpy as np
import torch
from schnetpack import AtomsLoader
from mlcalcdriver.globals import eVA
from mlcalcdriver.calculators import Calculator
from mlcalcdriver.calculators.utils import torch_derivative, get_derivative_names
from mlcalcdriver.interfaces import posinp_to_ase_atoms, SchnetPackData
from schnetpack.environment import SimpleEnvironmentProvider, AseEnvironmentProvider
from schnetpack.utils import load_model
from mlcalcdriver.globals import EV_TO_HA, B_TO_ANG


class SchnetPackCalculator(Calculator):
    r"""
    Calculator based on a SchnetPack model
    """

    def __init__(
        self,
        model_dir,
        args_dir=None,
        available_properties=None,
        device="cpu",
        units=eVA,
    ):
        r"""
        Parameters
        ----------
        model_path : str
            Path to the stored model on which the calculator
            will be based. If $MODELDIR is defined, the path can
            be relative to it. If not, the path must be absolute
            or relative to the working directory.
        available_properties : str or list of str
            Properties that the model can predict. If `None`, they
            automatically determined from the model. Default is `None`.
        device : str
            Can be either `"cpu"` to use cpu or `"cuda"` to use "gpu"
        """
        self.device = device
        try:
            self.model = load_model(model_dir, map_location=self.device)
        except Exception as e:
            self.model = load_model(
                os.environ["MODELDIR"] + model_dir, map_location=self.device
            )

        # Bugfix to make older models work with PyTorch 1.6
        # Hopefully temporary
        for mod in self.model.modules():
            if not hasattr(mod, "_non_persistent_buffers_set"):
                mod._non_persistent_buffers_set = set()
        self.model.requires_stress = False

        super(SchnetPackCalculator, self).__init__(units=units)
        self._get_representation_type()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = str(device).lower()

    def run(
        self, property, posinp=None, batch_size=128,
    ):
        r"""
        Main method to use when making a calculation with
        the calculator.
        """
        init_property, out_name, derivative, wrt = get_derivative_names(
            property, self.available_properties
        )

        if len(posinp) > 1 and derivative:
            batch_size = 1

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
        data_loader = AtomsLoader(data, batch_size=batch_size)

        pred = []
        if derivative == 0:
            if self.model.output_modules[0].derivative is not None:
                for batch in data_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    pred.append(self.model(batch))
            else:
                with torch.no_grad():
                    for batch in data_loader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        pred.append(self.model(batch))
        if abs(derivative) == 1:
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch[wrt[0]].requires_grad_()
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
        predictions = {}
        if derivative:
            predictions[property] = np.concatenate(
                [batch[out_name].cpu().detach().numpy() for batch in pred]
            )
        else:
            predictions[property] = np.concatenate(
                [batch[init_property].cpu().detach().numpy() for batch in pred]
            )
        return predictions

    def _get_available_properties(self):
        r"""
        Returns
        -------
        avail_prop
            Properties that the SchnetPack model can return
        """
        avail_prop = set()
        for out in self.model.output_modules:
            if out.derivative is not None:
                avail_prop.update([out.property, out.derivative])
            else:
                avail_prop.update([out.property])
        if "energy_U0" in avail_prop:
            avail_prop.add("energy")
        return list(avail_prop)

    def _get_representation_type(self):
        r"""
        Private method to determine representation type (schnet or wcasf).
        """
        if "representation.cutoff.cutoff" in self.model.state_dict().keys():
            self.model_type = "wacsf"
            self.cutoff = float(self.model.state_dict()["representation.cutoff.cutoff"])
        elif any(
            [
                name in self.model.state_dict().keys()
                for name in [
                    "module.representation.embedding.weight",
                    "representation.embedding.weight",
                ]
            ]
        ):
            self.model_type = "schnet"
            try:
                self.cutoff = float(
                    self.model.state_dict()[
                        "module.representation.interactions.0.cutoff_network.cutoff"
                    ]
                )
            except KeyError:
                self.cutoff = float(
                    self.model.state_dict()[
                        "representation.interactions.0.cutoff_network.cutoff"
                    ]
                )
        else:
            raise NotImplementedError("Model type is not recognized.")
