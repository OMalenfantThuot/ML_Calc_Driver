r"""
Calculator metaclass to accomodate machine learning models
trained using the SchnetPack package.
"""

import numpy as np
import torch
from schnetpack import AtomsLoader
from mlcalcdriver.globals import eVA
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.calculators.utils import torch_derivative, get_derivative_names
from mlcalcdriver.interfaces import posinp_to_ase_atoms, SchnetPackData, AtomsToPatches
from schnetpack.environment import SimpleEnvironmentProvider, AseEnvironmentProvider
from utils.models import PatchesAtomisticModel, PatchesAtomwise


class PatchSPCalculator(SchnetPackCalculator):
    r"""
    Calculator based on a SchnetPack model
    """

    def __init__(
        self,
        model_dir,
        available_properties=None,
        device="cpu",
        units=eVA,
        md=False,
        subgrid=None,
    ):
        super().__init__(
            model_dir=model_dir,
            available_properties=available_properties,
            device=device,
            units=units,
            md=md,
        )
        self.n_interaction = len(self.model.representation.interactions)
        self.subgrid = subgrid
        self._convert_model()

    @property
    def n_interaction(self):
        return self._n_interaction

    @n_interaction.setter
    def n_interaction(self, n_interaction):
        self._n_interaction = n_interaction

    @property
    def subgrid(self):
        return self._subgrid

    @subgrid.setter
    def subgrid(self, subgrid):
        if subgrid is None:
            self._subgrid = [1, 1, 1]
        else:
            assert len(subgrid) == 3
            self._subgrid = subgrid

    def run(
        self,
        property,
        posinp=None,
        batch_size=1,
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

        Returns
        -------
        predictions : :class:`numpy.ndarray`
            Corresponding prediction by the model.
        """

        # Initial setup
        assert (
            len(posinp) == 1
        ), "Use the PatchSPCalculator for one configuration at a time."
        atoms = posinp_to_ase_atoms(posinp[0])

        init_property, out_name, derivative, wrt = get_derivative_names(
            property, self.available_properties
        )
        pbc = True if atoms.pbc.any() else False
        environment_provider = (
            AseEnvironmentProvider(cutoff=self.cutoff)
            if pbc
            else SimpleEnvironmentProvider()
        )

        # Split the configuration according to the subgrid
        at_to_patches = AtomsToPatches(
            cutoff=self.cutoff, n_interaction=self.n_interaction, grid=self.subgrid
        )
        subcells, subcells_main_idx, original_cell_idx = at_to_patches.split_atoms(
            atoms
        )

        # Pass each subcell independantly
        results = []
        for subcell, main_idx in zip(subcells, subcells_main_idx):
            main_idx = torch.LongTensor(main_idx).to(self.device)
            data = SchnetPackData(
                data=[subcell],
                environment_provider=environment_provider,
                collect_triples=self.model_type == "wacsf",
            )
            data_loader = AtomsLoader(data, batch_size=1)

            if derivative == 0:
                if self.model.output_modules[0].derivative is not None:
                    print("test")
                    for batch in data_loader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        results.append(self.model(batch, main_idx))
                else:
                    with torch.no_grad():
                        for batch in data_loader:
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                            results.append(self.model(batch, main_idx))

            if abs(derivative) == 1:
                for batch in data_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    batch[wrt[0]].requires_grad_()
                    forward_results = self.model(batch, main_idx)
                    deriv1 = torch.unsqueeze(
                        torch_derivative(forward_results[init_property], batch[wrt[0]]),
                        0,
                    )
                    if derivative < 0:
                        deriv1 = -1.0 * deriv1
                    results.append({"forces": deriv1.squeeze()[main_idx]})

            if abs(derivative) == 2:
                raise NotImplementedError()

        predictions = {}
        if property == "energy":
            predictions["energy"] = np.sum(
                [patch["energy"].cpu().detach().numpy() for patch in results]
            )
        elif property == "forces":
            forces = np.concatenate(
                [patch["forces"].cpu().detach().numpy() for patch in results]
            )
            print(forces[0])
            idx = np.argsort(np.concatenate(subcells_main_idx))
            predictions["forces"] = forces[idx]

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

    def _convert_model(self):
        initout = self.model.output_modules[0]
        patches_output = PatchesAtomwise(initout.out_net[1].n_neurons[0])
        patches_output.load_state_dict(initout.state_dict())
        patches_model = PatchesAtomisticModel(self.model.representation, patches_output)
        self.model = patches_model
