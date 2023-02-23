from ase.geometry import Cell
from ase import Atom, Atoms
from collections.abc import Sequence
import numpy as np
from copy import deepcopy


class AtomsToPatches:
    """
    Splits an ase.Atoms into patches.
    Used by the PatchSPCalculator.
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

        # Fix for atoms exactly on the cell frontier
        new_scaled_positions = np.round(atoms.get_scaled_positions(), decimals=8)
        t_idx_0, t_idx_1 = np.where(
            np.isclose(new_scaled_positions, 1.0, rtol=0, atol=1e-8)
        )
        new_scaled_positions[t_idx_0, t_idx_1] -= 1.0

        new_pbc = []
        for i in range(3):
            new_pbc.append(True if atoms.pbc[i] and self.grid[i] == 1 else False)

        atoms = Atoms(symbols=atoms.symbols, cell=atoms.cell, pbc=new_pbc)
        atoms.set_scaled_positions(new_scaled_positions)

        # Define grid and cells
        full_cell = atoms.cell
        grid_cell = Cell(full_cell / np.broadcast_to(self.grid, (3, 3)).T)

        # Define buffers
        buffer_length = (self.cutoff * self.n_interaction) / np.sin(
            np.radians(min(full_cell.angles()))
        )
        full_scaled_buffer_length = buffer_length / full_cell.cellpar()[:3]
        full_scaled_buffer_length[np.where(self.grid == 1)[0]] = 0

        if np.any(full_scaled_buffer_length >= 0.5):
            raise ValueError("The supercell is too small to use with this buffer.")
        grid_scaled_buffer_length = full_scaled_buffer_length * self.grid
        if np.any(grid_scaled_buffer_length >= 1):
            raise ValueError("The grid is too fine to use with this buffer.")

        # Add initial buffer around the supercell
        buffered_atoms, copy_idx = add_initial_buffer(
            atoms, full_scaled_buffer_length, full_cell
        )

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
        main_subcell_idx_list = []
        original_atoms_idx_list = []
        complete_subcell_copy_idx_list = []

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

            complete_subcell_copy_idx = copy_idx[buffered_subcell_atoms_idx]

            main_subcell_idx = np.where(
                np.all(
                    np.floor(
                        np.round(
                            scaled_atoms_positions[buffered_subcell_atoms_idx],
                            decimals=8,
                        )
                    )
                    == subcells_idx[i],
                    axis=1,
                )
            )[0]

            subcell_as_atoms_list.append(buffered_atoms[buffered_subcell_atoms_idx])
            main_subcell_idx_list.append(main_subcell_idx)
            original_atoms_idx_list.append(buffered_subcell_atoms_idx[main_subcell_idx])
            complete_subcell_copy_idx_list.append(complete_subcell_copy_idx)

        # Returns:
        # 1) a list of atoms instances (subcells)
        # 2) a list of indexes of the atoms that
        #    are not in the buffer of those subcells
        # 3) a list of the original index of the atoms
        #    to map back per atom predicted properties
        #    to the original configuration.
        return (
            subcell_as_atoms_list,
            main_subcell_idx_list,
            original_atoms_idx_list,
            complete_subcell_copy_idx_list,
        )


def add_initial_buffer(atoms, scaled_buffer_length, full_cell):

    # Determine which atoms need to be copied
    init_scaled_positions = atoms.get_scaled_positions()
    in_buff_low = (init_scaled_positions < scaled_buffer_length).astype(int)
    in_buff_high = (init_scaled_positions > (1 - scaled_buffer_length)).astype(int)
    in_buff = in_buff_low - in_buff_high

    # Look at all possible permutations
    copy_idx = [i for i in range(len(atoms))]
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
                    copy_idx.append(i)
        if non_zero_dimensions >= 2:
            if x != 0:
                if y != 0:
                    atoms = copy_atom_with_translation(
                        atoms, i, np.array([x, y, 0]).dot(full_cell)
                    )
                    copy_idx.append(i)
                if z != 0:
                    atoms = copy_atom_with_translation(
                        atoms, i, np.array([x, 0, z]).dot(full_cell)
                    )
                    copy_idx.append(i)
            else:
                atoms = copy_atom_with_translation(
                    atoms, i, np.array([0, y, z]).dot(full_cell)
                )
                copy_idx.append(i)
        if non_zero_dimensions == 3:
            atoms = copy_atom_with_translation(
                atoms, i, np.array([x, y, z]).dot(full_cell)
            )
            copy_idx.append(i)

    return atoms, np.array(copy_idx)


def copy_atom_with_translation(atoms, idx, translation):
    # Add atom to existing configuration
    new_atom = Atom(atoms[idx].symbol, atoms[idx].position + translation)
    atoms.append(new_atom)
    return atoms
