import torch
import numpy as np
from torch.utils.data import Dataset
from schnetpack.data.atoms import _convert_atoms, torchify_dict


class SchnetPackData(Dataset):
    r"""
    Class used to interface data from the mlcalcdriver package
    as a PyTorch Dataset understood by SchnetPack.
    """

    def __init__(self, data, environment_provider, collect_triples=False):
        self.data = data
        self.environment_provider = environment_provider
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
        properties = _convert_atoms(
            at,
            environment_provider=self.environment_provider,
            collect_triples=self.collect_triples,
        )
        return at, properties
