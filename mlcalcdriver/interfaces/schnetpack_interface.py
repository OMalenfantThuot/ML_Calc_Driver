import torch
import numpy as np
from torch.utils.data import Dataset
from schnetpack.data.atoms import _convert_atoms, torchify_dict
from schnetpack import Properties


class SchnetPackData(Dataset):
    r"""
    Class used to interface data from the mlcalcdriver package
    as a PyTorch Dataset understood by SchnetPack.
    """

    def __init__(
        self, data, environment_provider, atomic_environment=None, collect_triples=False
    ):
        self.data = data
        self.environment_provider = environment_provider
        self.atomic_environment = atomic_environment
        self.collect_triples = collect_triples

    def __len__(self):
        r"""
        Needed to create a PyTorch dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        r"""
        Needed to create a PyTorch Dataset
        """
        _, properties = self.get_properties(idx)
        properties["_idx"] = np.array([idx], dtype=int)
        return torchify_dict(properties)

    def get_properties(self, idx):
        r"""
        Returns property dictionary at given index.

        Parameters
        ----------
        idx : int

        Returns
        -------
        at : :class:`ase.Atoms`
        properties : dict
        """
        idx = int(idx)
        at = self.data[idx]

        # extract/calculate structure
        if self.atomic_environment is None:
            properties = _convert_atoms(
                at,
                environment_provider=self.environment_provider,
                collect_triples=self.collect_triples,
            )
        else:
            properties = fast_convert(at, atomic_environment=self.atomic_environment)
        return at, properties


def fast_convert(atoms, atomic_environment):
    inputs = {}
    inputs[Properties.Z] = atoms.numbers.astype(int)
    inputs[Properties.R] = atoms.positions.astype(np.float32)

    nbh_idx, offsets = atomic_environment
    inputs[Properties.neighbors] = nbh_idx.astype(int)

    inputs[Properties.cell] = np.array(atoms.cell.array, dtype=np.float32)
    inputs[Properties.cell_offset] = offsets.astype(np.float32)
    return inputs
