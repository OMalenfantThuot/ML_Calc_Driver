from ase.geometry import Cell
from ase import Atom
from collections.abc import Sequence
import numpy as np
from copy import deepcopy


class AtomsToPatches:
    """
    Splits an ase.Atoms into patches.
    """

    def __init__(self, cutoff, n_interaction, grid):
        self.cutoff = cutoff
        self.n_interaction = n_interaction
        self.grid = grid

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

    def split_atoms(self, atoms):
        # Define grid and cells
        atoms = deepcopy(atoms)
        atoms.set_pbc(self.grid == 1)
        full_cell = atoms.cell
        grid_cell = Cell(full_cell / np.broadcast_to(self.grid, (3, 3)).T)

        # Define buffers
        buffer_length = (self.cutoff * self.n_interaction) / np.sin(
            np.radians(min(full_cell.angles()))
        )
        full_scaled_buffer_length = buffer_length / full_cell.cellpar()[:3]
        full_scaled_buffer_length[np.where(atoms.pbc)[0]] = 0
        if np.any(full_scaled_buffer_length >= 0.5):
            raise ValueError("The supercell is too small to use with this buffer.")
        grid_scaled_buffer_length = full_scaled_buffer_length * self.grid
        if np.any(grid_scaled_buffer_length >= 1):
            raise ValueError("The grid is too fine to use with this buffer.")

        # Add initial buffer around the supercell
        buffered_atoms = add_initial_buffer(atoms, full_scaled_buffer_length, full_cell)

        # Define grid indexes
        dim0, dim1, dim2 = (
            np.linspace(0, self.grid[0] - 1, self.grid[0]),
            np.linspace(0, self.grid[1] - 1, self.grid[1]),
            np.linspace(0, self.grid[2] - 1, self.grid[2]),
        )
        gridx, gridy, gridz = np.meshgrid(dim0, dim1, dim2)
        subcells_idx = np.concatenate(
            (gridx.reshape(-1, 1), gridy.reshape(-1, 1), gridz.reshape(-1, 1)), axis=1
        )

        # Scaling atomic positions in grid units
        scaled_atoms_positions = buffered_atoms.get_scaled_positions() * self.grid

        # Create subcells as atoms instances
        subcell_as_atoms_list = []
        main_atoms_idx_list = []
        for i, subcell in enumerate(subcells_idx):
            buffered_subcell_min = subcell - grid_scaled_buffer_length
            buffered_subcell_max = subcell + 1 + grid_scaled_buffer_length

            buffered_subcell_atoms_idx = np.where(
                np.all(
                    np.logical_and(
                        scaled_atoms_positions >= buffered_subcell_min,
                        scaled_atoms_positions < buffered_subcell_max,
                    ),
                    axis=1,
                )
            )[0]

            main_subcell_idx = np.where(
                np.all(
                    np.floor(
                        np.around(
                            scaled_atoms_positions[buffered_subcell_atoms_idx],
                            decimals=8,
                        )
                    )
                    == subcells_idx[i],
                    axis=1,
                )
            )[0]

            subcell_as_atoms_list.append(
                deepcopy(buffered_atoms[buffered_subcell_atoms_idx])
            )
            main_atoms_idx_list.append(main_subcell_idx)

        # Return a list of atoms instances and a list of indexes of
        # the atoms that are not in the buffer of those subcells
        return subcell_as_atoms_list, main_atoms_idx_list


def add_initial_buffer(atoms, scaled_buffer_length, full_cell):
    init_scaled_positions = atoms.get_scaled_positions()
    in_buff_low = (init_scaled_positions < scaled_buffer_length).astype(int)
    in_buff_high = (init_scaled_positions > (1 - scaled_buffer_length)).astype(int)
    in_buff = in_buff_low - in_buff_high

    for i in range(init_scaled_positions.shape[0]):
        non_zero_dimensions = np.sum(np.absolute(in_buff[i]))
        x, y, z = in_buff[i]
        if non_zero_dimensions == 0:
            pass
        if non_zero_dimensions >= 1:
            for dim, translation in zip(
                [x, y, z],
                [
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                ],
            ):
                if dim != 0:
                    atoms = copy_atom_with_translation(
                        atoms, i, (translation * dim).dot(full_cell)
                    )
        if non_zero_dimensions >= 2:
            if x != 0:
                if y != 0:
                    atoms = copy_atom_with_translation(
                        atoms, i, np.array([x, y, 0]).dot(full_cell)
                    )
                if z != 0:
                    atoms = copy_atom_with_translation(
                        atoms, i, np.array([x, 0, z]).dot(full_cell)
                    )
            else:
                atoms = copy_atom_with_translation(
                    atoms, i, np.array([0, y, z]).dot(full_cell)
                )
        if non_zero_dimensions == 3:
            atoms = copy_atom_with_translation(
                atoms, i, np.array([x, y, z]).dot(full_cell)
            )

    return atoms


def copy_atom_with_translation(atoms, idx, translation):
    new_atom = Atom(atoms[idx].symbol, atoms[idx].position + translation)
    atoms.append(new_atom)
    return atoms
