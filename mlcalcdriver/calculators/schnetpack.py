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
from mlcalcdriver.interfaces import posinp_to_ase_atoms, SchnetPackData
from schnetpack.environment import SimpleEnvironmentProvider, AseEnvironmentProvider


class SchnetPackCalculator(Calculator):
    r"""
    Calculator based on a SchnetPack model
    """

    def __init__(self, model_dir, available_properties=None, device="cpu", units=eVA):
        r"""
        Parameters
        ----------
        model_dir : str
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
        try:
            self.model = load_model(
                model_dir=os.environ["MODELDIR"] + model_dir, device=device
            )
        except Exception:
            self.model = load_model(model_dir=model_dir, device=device)
        super(SchnetPackCalculator, self).__init__(units=units)
        self._get_representation_type()

    def run(
        self, property, derivative=False, posinp=None, device="cpu", batch_size=128,
    ):
        r"""
        Main method to use when making a calculation with
        the calculator.
        """
        if property not in self.available_properties:
            if derivative:
                if property == "forces":
                    if "energy" in self.available_properties:
                        init_property, deriv_name, out_name = (
                            "energy",
                            "forces",
                            "forces",
                        )
                    else:
                        raise ValueError(
                            "This model can't be used for forces predictions."
                        )
                else:
                    raise NotImplementedError(
                        "Derivatives of other quantities than the energy are not implemented yet."
                    )
            else:
                raise ValueError(
                    "The property {} is not in the available properties of the model : {}.".format(
                        property, self.available_properties
                    )
                )
        elif property == "energy" and "energy_U0" in self.available_properties:
            init_property, out_name = "energy_U0", "energy"
        else:
            init_property, out_name = property, property

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
        if self.model.output_modules[0].derivative is not None:
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred.append(self.model(batch))
        elif derivative:
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch["_positions"].requires_grad_()
                results = self.model(batch)
                #print(results[init_property])
                #print(torch.ones_like(results[init_property]))
                drdx = (
                    -1.0
                     *  torch.autograd.grad(
                        results[init_property],
                        batch["_positions"],
                        grad_outputs=torch.ones_like(results[init_property]),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                )
                pred.append({deriv_name: drdx})
            if derivative == 2:
                hessian = np.zeros([6, 6])
                print(drdx)
                print(batch["_positions"])
                k=0
                for i in range(drdx.shape[2]):
                    for j in range(drdx.shape[1]):
                        print(drdx[0,j,i])
                        drdx2= torch.autograd.grad(
                                drdx[0, j, i],
                                batch["_positions"],
                                grad_outputs=torch.ones_like(drdx[0,j,i]),
                                create_graph=True,
                                retain_graph=True,
                                #allow_unused=True,
                            )[0]
                        print(drdx2)
                        print(k)
                        hessian[k] = drdx2.detach().numpy().flatten()
                        k += 1
                print(hessian)
                hessian = (hessian + hessian.T)/2
                print(np.around(hessian, decimals=2))
        else:
            with torch.no_grad():
                pred = []
                for batch in data_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pred.append(self.model(batch))

        predictions = {}
        if derivative:
            predictions[out_name] = np.concatenate(
                [batch[deriv_name].cpu().detach().numpy() for batch in pred]
            )
        else:
            predictions[out_name] = np.concatenate(
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


def load_model(model_dir, device):
    try:
        model_dir = str(model_dir)
    except Exception as e:
        raise e
    try:
        model = torch.load(model_dir, map_location=device)
    except Exception as e:
        raise e
    return model
