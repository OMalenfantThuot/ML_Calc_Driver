r"""
Calculator subclass to accomodate machine learning models
trained using the SchnetPack package and configurations
split over individual patches.
"""

import numpy as np
import scipy
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
        sparse=False,
        atomic_environments=None,
        patches=None,
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
        self.atomic_environments = atomic_environments
        self.patches = patches
        self.sparse = sparse
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
            self._subgrid = np.array(subgrid)

    @property
    def atomic_environments(self):
        return self._atomic_environments

    @atomic_environments.setter
    def atomic_environments(self, atomic_environments):
        if atomic_environments is not None:
            assert len(atomic_environments) == np.prod(self.subgrid)
            self._atomic_environments = atomic_environments
        else:
            self._atomic_environments = [None] * np.prod(self.subgrid)

    @property
    def patches(self):
        return self._patches

    @patches.setter
    def patches(self, patches):
        if patches is not None:
            for el in patches:
                assert len(el) == np.prod(self.subgrid)
            self._patches = patches
        else:
            self._patches = None

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
        if self.patches is None:
            at_to_patches = AtomsToPatches(
                cutoff=self.cutoff, n_interaction=self.n_interaction, grid=self.subgrid
            )
            (
                subcells,
                subcells_main_idx,
                original_cell_idx,
                complete_subcell_copy_idx,
            ) = at_to_patches.split_atoms(atoms)
        else:
            (
                subcells,
                subcells_main_idx,
                original_cell_idx,
                complete_subcell_copy_idx,
            ) = self.patches

        # Pass each subcell independantly
        results = []
        for subcell, env in zip(subcells, self.atomic_environments):
            data = SchnetPackData(
                data=[subcell],
                environment_provider=environment_provider,
                atomic_environment=env,
                collect_triples=self.model_type == "wacsf",
            )
            data_loader = AtomsLoader(data, batch_size=1)
            if derivative == 0:
                if self.model.output_modules[0].derivative is not None:
                    for batch in data_loader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        patch_results = self.model(batch)
                        cpu_patch_results = {}
                        for key, value in patch_results.items():
                            cpu_patch_results[key] = value.detach().cpu().numpy()
                            del value
                        results.append(cpu_patch_results)
                        del patch_results
                else:
                    with torch.no_grad():
                        for batch in data_loader:
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                            patch_results = self.model(batch)
                            cpu_patch_results = {}
                            for key, value in patch_results.items():
                                cpu_patch_results[key] = value.detach().cpu().numpy()
                                del value
                            results.append(cpu_patch_results)
                            del patch_results

            if abs(derivative) == 1:
                for batch in data_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    batch[wrt[0]].requires_grad_()
                    patch_forward_results = self.model(batch)
                    patch_deriv1 = torch_derivative(
                        patch_forward_results[init_property], batch[wrt[0]]
                    )
                    if derivative < 0:
                        patch_deriv1 = -1.0 * patch_deriv1
                    cpu_patch_deriv1 = patch_deriv1.detach().cpu().numpy()
                    results.append({out_name: cpu_patch_deriv1})
                    for key, value in patch_forward_results.items():
                        del value
                    del patch_forward_results, patch_deriv1

            if abs(derivative) == 2:
                raise NotImplementedError()

        predictions = {}
        if property == "energy":
            predictions["energy"] = np.sum(
                [
                    patch["individual_energy"][subcells_main_idx[i]]
                    for i, patch in enumerate(results)
                ]
            )

        elif property == "forces":
            forces = np.zeros((len(atoms), 3), dtype=np.float32)
            for i in range(len(results)):
                forces[original_cell_idx[i]] = results[i]["forces"].squeeze()[
                    subcells_main_idx[i]
                ]
            predictions["forces"] = forces

        elif property == "hessian":
            hess_shape = (3 * len(atoms), 3 * len(atoms))
            if self.sparse:
                data_lims = [
                    9 * s.size * cs.size
                    for (s, cs) in zip(subcells_main_idx, complete_subcell_copy_idx)
                ]
                data_lims.insert(0, 0)
                data_lims = np.cumsum(data_lims)
                num_data = data_lims[-1]

                data = np.zeros(num_data, dtype=np.float32)
                row, col = np.zeros(num_data, dtype=np.intc), np.zeros(
                    num_data, dtype=np.intc
                )
            else:
                hessian = np.zeros(hess_shape, dtype=np.float32)

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

                if self.sparse:
                    row[data_lims[i] : data_lims[i + 1]] = (
                        hessian_original_cell_idx_0.flatten()
                    )
                    col[data_lims[i] : data_lims[i + 1]] = (
                        hessian_original_cell_idx_1.flatten()
                    )
                    data[data_lims[i] : data_lims[i + 1]] = (
                        results[i]["hessian"]
                        .copy()
                        .squeeze()[
                            hessian_subcells_main_idx_0, hessian_subcells_main_idx_1
                        ]
                        .flatten()
                    )
                else:
                    hessian[
                        hessian_original_cell_idx_0, hessian_original_cell_idx_1
                    ] = results[i]["hessian"].squeeze()[
                        hessian_subcells_main_idx_0, hessian_subcells_main_idx_1
                    ]
                del hessian_subcells_main_idx_0, hessian_subcells_main_idx_1
                del hessian_original_cell_idx_0, hessian_original_cell_idx_1

            if self.sparse:
                hessian = scipy.sparse.coo_array(
                    (data, (row, col)), shape=hess_shape, dtype=np.float32
                )
                hessian.eliminate_zeros()
                hessian = hessian.tocsr()
            else:
                hessian = np.expand_dims(hessian, 0)

            predictions["hessian"] = hessian

        else:
            raise NotImplementedError()

        return predictions

    def _convert_model(self):
        from utils.models import PatchesAtomisticModel, PatchesAtomwise

        initout = self.model.output_modules[0]
        aggregation_mode = "mean" if initout.atom_pool.average else "sum"
        atomref = (
            initout.atomref.weight.cpu().numpy()
            if initout.atomref is not None
            else None
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
            atomref=atomref,
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
    return idx_0.astype(np.intc), idx_1.astype(np.intc)


def collect_results(patch_results):
    cpu_patch_results = {}
    for key, value in patch_results.items():
        cpu_patch_results[key] = value.detach().cpu().numpy()
        del value
    del patch_results
    return cpu_patch_results
