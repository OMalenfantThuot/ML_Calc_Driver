from ase.geometry import Cell
from collections.abc import Sequence
import numpy as np


class AtomsToPatches:
    """ """

    def __init__(self, cutoff, n_interaction, grid, dimensions=2):
        self.cutoff = cutoff
        self.n_interaction = n_interaction
        self.grid = grid
        self.dimensions = dimensions

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, cutoff):
        self._cutoff = float(cutoff)

    @property
    def n_interaction(self):
        return self._n_interaction

    @n_interaction.setter
    def n_interaction(self, n_interaction):
        assert isinstance(
            n_interaction, int
        ), "The number of interaction blocks should be an integer."
        self._n_interaction = n_interaction

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        if isinstance(grid, np.ndarray):
            assert grid.shape == (
                3,
            ), "The grid given to the EnvironmentProvider is not valid."
        elif not isinstance(grid, Sequence):
            raise TypeError(
                "The grid should be given as a numpy.ndarray, or a Sequence."
            )
        else:
            assert (
                len(grid) == 3
            ), "The grid given to the EnvironmentProvider is not valid."
        self._grid = np.array(grid)

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions):
        self._dimensions = int(dimensions)

    def split_atoms(self, atoms):
        full_cell = atoms.cell
        grid_cell = Cell(full_cell / np.broadcast_to(self.grid, (3, 3)).T)

        buffer_length = self.cutoff * self.n_interaction
        scaled_buffer_length = buffer_length / grid_cell.cellpar()[:3]

        dim0, dim1, dim2 = (
            np.linspace(0, self.grid[0] - 1, self.grid[0]),
            np.linspace(0, self.grid[1] - 1, self.grid[1]),
            np.linspace(0, self.grid[2] - 1, self.grid[2]),
        )
        gridx, gridy, gridz = np.meshgrid(dim0, dim1, dim2)
        subcells_idx = np.concatenate(
            (gridx.reshape(-1, 1), gridy.reshape(-1, 1), gridz.reshape(-1, 1)), axis=1
        )

        scaled_atoms_positions = atoms.get_scaled_positions() * self.grid

        atoms_dict = {}
        for i in range(len(subcells_idx)):
            atoms_dict[i] = []

        for i, atom_position in enumerate(scaled_atoms_positions):
            center_subcell = int(
                np.where(
                    np.all(
                        np.logical_and(
                            atom_position >= subcells_idx,
                            atom_position < subcells_idx + 1,
                        ),
                        axis=1,
                    )
                )[0]
            )
            atoms_dict[center_subcell].append(atoms[i])
