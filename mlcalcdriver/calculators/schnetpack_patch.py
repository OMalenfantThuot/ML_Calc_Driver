r"""
Calculator subclass to accomodate machine learning models
trained using the SchnetPack package and configurations
split over individual patches.
"""

import numpy as np
import torch
import warnings
from schnetpack import AtomsLoader
from mlcalcdriver.globals import eVA
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.calculators.utils import torch_derivative, get_derivative_names
from mlcalcdriver.interfaces import posinp_to_ase_atoms, SchnetPackData, AtomsToPatches
from schnetpack.environment import SimpleEnvironmentProvider, AseEnvironmentProvider


class PatchSPCalculator(SchnetPackCalculator):
    r"""
    Calculator based on a SchnetPack model

    Parameters
    ----------
    model_dir : str
        Path to the stored model.
    available_properties : str or list of str
        Same as SchnetPackCalculator
    device : str
        Same as SchnetPackCalculator
    units : dict
        Same as SchnetPackCalculator
    md : bool
        Same as SchnetPackCalculator
    subgrid : :class:`Sequence` of length 3
        Number of subdivisions of the initial configuration in 
        all 3 dimensions. The periodic boundary conditions will
        be kept in the dimensions with 1.
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

        if property == "hessian" and any(self.subgrid == 2):
            raise warnings.warn(
                """
            The hessian matrix can have some bad values with a grid of
            size 2 because the same atom can be copied multiple times
            in the buffers of the same subcell. Use a larger grid.
            """
            )

        init_property, out_name, derivative, wrt = get_derivative_names(
            property, self.available_properties
        )
        if abs(derivative) >= 1:
            self.model.output_modules[0].create_graph = True

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
        (
            subcells,
            subcells_main_idx,
            original_cell_idx,
            complete_subcell_copy_idx,
        ) = at_to_patches.split_atoms(atoms)

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
            hessian = np.zeros((3 * len(atoms), 3 * len(atoms)))

            for i in range(len(results)):

                (
                    hessian_original_cell_idx_0,
                    hessian_original_cell_idx_1,
                ) = prepare_hessian_indices(
                    original_cell_idx[i], complete_subcell_copy_idx[i]
                )

                (
                    hessian_subcells_main_idx_0,
                    hessian_subcells_main_idx_1,
                ) = prepare_hessian_indices(
                    subcells_main_idx[i],
                    np.arange(0, len(complete_subcell_copy_idx[i])),
                )

                hessian[hessian_original_cell_idx_0, hessian_original_cell_idx_1] = (
                    results[i]["hessian"]
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy()[hessian_subcells_main_idx_0, hessian_subcells_main_idx_1]
                )
            predictions["hessian"] = hessian

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


def prepare_hessian_indices(input_idx_0, input_idx_1):

    bias_0 = np.tile(np.array([0, 1, 2]), len(input_idx_0))
    bias_1 = np.tile(np.array([0, 1, 2]), len(input_idx_1))
    hessian_idx_0 = np.repeat(3 * input_idx_0, 3) + bias_0
    hessian_idx_1 = np.repeat(3 * input_idx_1, 3) + bias_1
    idx_0, idx_1 = np.meshgrid(hessian_idx_0, hessian_idx_1, indexing="ij")
    return idx_0, idx_1
