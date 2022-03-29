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
        for subcell in subcells:
            data = SchnetPackData(
                data=[subcell],
                environment_provider=environment_provider,
                collect_triples=self.model_type == "wacsf",
            )
            data_loader = AtomsLoader(data, batch_size=1)

            if derivative == 0:
                if self.model.output_modules[0].derivative is not None:
                    for batch in data_loader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        results.append(self.model(batch))
                else:
                    with torch.no_grad():
                        for batch in data_loader:
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                            results.append(self.model(batch))

            if abs(derivative) == 1:
                for batch in data_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    batch[wrt[0]].requires_grad_()
                    forward_results = self.model(batch)
                    deriv1 = torch_derivative(
                        forward_results[init_property], batch[wrt[0]]
                    )
                    if derivative < 0:
                        deriv1 = -1.0 * deriv1
                    results.append({out_name: deriv1})

            if abs(derivative) == 2:
                raise NotImplementedError()

        predictions = {}
        if property == "energy":
            predictions["energy"] = np.sum(
                [
                    patch["individual_energy"][subcells_main_idx[i]]
                    .detach()
                    .cpu()
                    .numpy()
                    for i, patch in enumerate(results)
                ]
            )
        elif property == "forces":
            forces = np.zeros((len(atoms), 3))
            for i in range(len(results)):
                forces[original_cell_idx[i]] = (
                    results[i]["forces"]
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy()[subcells_main_idx[i]]
                )
            predictions["forces"] = forces
        elif property == "hessian":
            print(results[0]["hessian"].shape)
        else:
            raise NotImplementedError()

        return predictions

    def _convert_model(self):
        from utils.models import PatchesAtomisticModel, PatchesAtomwise

        initout = self.model.output_modules[0]
        aggregation_mode = "mean" if initout.atom_pool.average else "sum"
        atomref = (
            initout.atomref.weight.numpy() if initout.atomref is not None else None
        )

        patches_output = PatchesAtomwise(
            n_in=initout.out_net[1].n_neurons[0],
            n_out=initout.out_net[1].n_neurons[-1],
            aggregation_mode=aggregation_mode,
            n_layers=initout.n_layers,
            property=initout.property,
            contributions=initout.contributions,
            derivative=initout.derivative,
            negative_dr=initout.negative_dr,
            stress=initout.stress,
            create_graph=initout.create_graph,
            atomref=initout.atomref,
        )
        patches_output.load_state_dict(initout.state_dict())

        patches_model = PatchesAtomisticModel(self.model.representation, patches_output)
        self.model = patches_model.to(self.device)
