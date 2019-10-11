import torch
import numpy as np
from torch.utils.data import Dataset
from schnetpack.data.atoms import _convert_atoms

class SchnetPackData(Dataset):
    r"""
    Class used to interface data from the mlcalcdriver package
    as a PyTorch Dataset understood by SchnetPack.
    """

    def __init__(
        self,
        data,
        environment_provider,
        collect_triples=False,
        center_positions=False,
    ):
        self.data = data
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.center_positions = center_positions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, properties = self.get_properties(idx)
        properties["_idx"] = torch.LongTensor(np.array([idx], dtype=np.int))
        return properties

    def get_properties(self, idx):
        """
        Return property dictionary at given index.

        Args:
            idx (int): data index

        Returns:
            at : ase.Atoms object
            properties (dict) : inputs formatted for SchnetPack
        """
        idx = int(idx)
        at = self.data[idx]

        # extract/calculate structure
        properties = _convert_atoms(
            at,
            environment_provider=self.environment_provider,
            collect_triples=self.collect_triples,
            center_positions=self.center_positions,
        )
        return at, properties
