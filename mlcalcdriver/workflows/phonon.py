r"""
The :class:`Phonon` class allows to compute the normal modes
and vibration energies of a system using a machine
learning trained model.
"""

import numpy as np
from mlcalcdriver import Job, Posinp
from mlcalcdriver.calculators import Calculator
from copy import deepcopy
from mlcalcdriver.globals import ANG_TO_B, B_TO_ANG, EV_TO_HA, HA_TO_CMM1, AMU_TO_EMU


class Phonon:
    r"""
    This class allows to run all the calculations enabling the
    computation of the phonon energies of a given system, using
    machine learning models.

    To get the phonon energies of the system, one needs to find the
    eigenvalues of the dynamical matrix, that is closely related to the
    Hessian matrix. To build these matrices, one must find the
    derivatives of the forces when each coordinate of each atom is
    translated by a small amount around the equilibrium positions.
    """

    def __init__(
        self,
        posinp,
        calculator,
        relax=False,
        finite_difference=False,
        translation_amplitudes=None,
    ):
        r"""
        The initial position fo the atoms are taken from the `init_state`
        Posinp instance. If they are not part of a relaxed geometry, the
        relax parameter should stay at `True`.

        WARNING: Relaxed geometries are dependent on the model chosen to
        define the calculator. In doubt, `relax` parameter should be ignored.

        The distance of the displacement in each direction is controlled
        by `translation_amplitudes`.

        Phonon energies and normal modes are calculated using the `run()`method.
        This method creates the additional structures needed, passes them to a
        `Job` instance, then post-processes the obtained forces
        to obtain them.

        Parameters
        ----------
        posinp : mlcaldriver.Posinp
            Initial positions of the system under consideration.
        calculator : Calculator
            mlcalcdriver.Calculator instance that will be used in
            the created Jobs to evaluate properties.
        relax : bool
            Wether the initial positions need to be relaxed or not.
            Default is `False`.
        finite_difference: bool
            If True, the hessian matrix is calculated using finite
            displacements of atoms. Default is False. Mostly there for
            legacy reasons.
        translation_amplitudes: list of length 3
            Amplitudes of the translations to be applied to each atom
            along each of the three space coordinates (in angstroms).
            Only relevant if finite_difference is True.
        """
        self.posinp = posinp
        self.calculator = calculator
        self.relax = relax
        self.finite_difference = finite_difference
        self.translation_amplitudes = translation_amplitudes

        if self.relax:
            self._ground_state = None
        else:
            self._ground_state = deepcopy(self.posinp)

        self.dyn_mat = None
        self.energies = None
        self.normal_modes = None

    @property
    def posinp(self):
        r"""
        Returns
        -------
        posinp : Posinp
            Initial positions of the system for which phonon properties
            will be calculated.
        """
        return self._posinp

    @posinp.setter
    def posinp(self, posinp):
        if isinstance(posinp, Posinp):
            self._posinp = posinp
        else:
            raise TypeError(
                "Initial positions should be given in a mlcalcdriver.Posinp instance."
            )

    @property
    def calculator(self):
        r"""
        Returns
        -------
        Calculator
            The Calculator object to use for the Jobs necessary to
            perform the phonons calculations.
        """
        return self._calculator

    @calculator.setter
    def calculator(self, calculator):
        if isinstance(calculator, Calculator):
            self._calculator = calculator
        else:
            raise TypeError(
                """
                The calculator for the Phonon instance must be a class or a
                metaclass derived from mlcalcdriver.calculators.Calculator.
                """
            )

    @property
    def translation_amplitudes(self):
        r"""
        Returns
        -------
        translation_amplitudes : float
            Displacements of atoms in all three dimensions to calculate
            the phonon properties. Default is 0.03 Angstroms.
        """
        return self._translation_amplitudes

    @translation_amplitudes.setter
    def translation_amplitudes(self, translation_amplitudes):
        if translation_amplitudes is None:
            self._translation_amplitudes = 0.03
        else:
            self._translation_amplitudes = float(translation_amplitudes)

    @property
    def relax(self):
        r"""
        Returns
        -------
        relax : bool
            If `True`, the initial positions are relaxed
            before the phonon properties are calculated.
            Recommended when comparing models.
        """
        return self._relax

    @relax.setter
    def relax(self, relax):
        relax = bool(relax)
        self._relax = relax

    @property
    def finite_difference(self):
        r"""
        Returns
        -------
        finite_difference : bool
            If `True`, the hessian matrix is calculated using small finite
            movements on the atoms. Default is `False`.
        """
        return self._finite_difference

    @finite_difference.setter
    def finite_difference(self, finite_difference):
        self._finite_difference = bool(finite_difference)

    @property
    def energies(self):
        r"""
        Returns
        -------
        numpy.array or None
            Phonon energies of the system (units: cm^-1).
        """
        return self._energies

    @energies.setter
    def energies(self, energies):
        self._energies = energies

    @property
    def dyn_mat(self):
        r"""
        Returns
        -------
        numpy.array or None
            Dynamical matrix deduced from the calculations.
        """
        return self._dyn_mat

    @dyn_mat.setter
    def dyn_mat(self, dyn_mat):
        self._dyn_mat = dyn_mat

    @property
    def normal_modes(self):
        r"""
        Returns
        -------
        numpy.array or None
            Normal modes of the system found as eigenvectors of the
            dynamical matrix.
        """
        return self._normal_modes

    @normal_modes.setter
    def normal_modes(self, normal_modes):
        self._normal_modes = normal_modes

    def run(self, batch_size=1, **kwargs):
        r"""
        Parameters
        ----------
        batch_size : int
            Batch size used when passing the structures to the model
        **kwargs :
            Optional arguments for the geometry optimization.
            Only useful if the relaxation is unstable.
        """
        if self.relax:
            from mlcalcdriver.workflows import Geopt

            geopt = Geopt(posinp=self.posinp, calculator=self.calculator, **kwargs)
            geopt.run(batch_size=batch_size)
            self._ground_state = deepcopy(geopt.final_posinp)

        if self.finite_difference:
            job = Job(posinp=self._create_displacements(), calculator=self.calculator)
            job.run(property="forces", batch_size=batch_size)
        else:
            job = Job(posinp=self._ground_state, calculator=self.calculator)
            job.run(property="hessian", batch_size=batch_size)
        self._post_proc(job)

    def _create_displacements(self):
        r"""
        Set the displacements each atom must undergo from the amplitudes
        of displacement in each direction. The numerical derivatives are obtained
        with the five-point stencil method. Only used if the phonons are
        calculated using finite_displacements.
        """
        structs = []
        for i in range(len(self._ground_state)):
            for dim in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
                for factor in [2, 1, -1, -2]:
                    structs.append(
                        self._ground_state.translate_atom(
                            i, self.translation_amplitudes * dim * factor
                        )
                    )
        return structs

    def _post_proc(self, job):
        r"""
        Calculates the energies and normal modes from the results
        obtained from the model.
        """
        self.dyn_mat = self._compute_dyn_mat(job)
        self.energies, self.normal_modes = self._solve_dyn_mat()
        self.energies *= HA_TO_CMM1

    def _compute_dyn_mat(self, job):
        r"""
        Computes the dynamical matrix
        """
        hessian = self._compute_hessian(job)
        masses = self._compute_masses()
        return hessian / masses

    def _compute_masses(self):
        r"""
        Creates the masses matrix
        """
        to_mesh = [atom.mass for atom in self._ground_state for _ in range(3)]
        m_i, m_j = np.meshgrid(to_mesh, to_mesh)
        return np.sqrt(m_i * m_j) * AMU_TO_EMU

    def _compute_hessian(self, job):
        r"""
        Computes the hessian matrix from the forces
        """
        n_at = len(self.posinp)
        if "hessian" in job.results.keys():
            h = (
                job.results["hessian"].reshape(3 * n_at, 3 * n_at)
                * EV_TO_HA
                * B_TO_ANG**2
            )
            return (h + h.T) / 2.0
        else:
            hessian = np.zeros((3 * n_at, 3 * n_at))
            forces = np.array(job.results["forces"]) * EV_TO_HA * B_TO_ANG
            for i in range(3 * n_at):
                hessian[i, :] = (
                    -forces[4 * i].flatten()
                    + forces[4 * i + 3].flatten()
                    + 8 * (forces[4 * i + 1].flatten() - forces[4 * i + 2].flatten())
                ) / (12 * self.translation_amplitudes * ANG_TO_B)
        return -(hessian + hessian.T) / 2.0

    def _solve_dyn_mat(self):
        r"""
        Obtains the eigenvalues and eigenvectors from
        the dynamical matrix
        """
        eigs, vecs = np.linalg.eigh(self.dyn_mat)
        eigs = np.sign(eigs) * np.sqrt(np.where(eigs < 0, -eigs, eigs))
        return eigs, vecs


class LanczosPhonon(Phonon):
    def __init__(self, posinp, calculator, relax=False):
        super().__init__(
            posinp=posinp, calculator=calculator, relax=relax, finite_difference=False
        )

    def run(self, batch_size=1, displacement=0.005, num_eval=4, **kwargs):
        r""" """
        from scipy.linalg import eigh_tridiagonal

        if self.relax:
            from mlcalcdriver.workflows import Geopt

            geopt = Geopt(posinp=self.posinp, calculator=self.calculator, **kwargs)
            geopt.run(batch_size=batch_size)
            self._ground_state = deepcopy(geopt.final_posinp)

        n_atoms = len(self._ground_state)

        projection_function = self._get_projection_function(num_eval)
        factors = self._get_factors(num_eval)

        # Empty matrices
        V_matrix = np.zeros((3 * n_atoms, 3 * n_atoms))
        T_diag = np.zeros(3 * n_atoms)
        T_offdiag = np.zeros(3 * n_atoms - 1)

        # Initial values
        v = np.ones(3 * n_atoms) / np.sqrt(3 * n_atoms)
        v = np.random.rand(3 * n_atoms)
        v = v / np.linalg.norm(v)
        beta, oldv = 0, 0

        # Lanczos iterations
        for it in range(3 * n_atoms):
            if it > 0:
                oldv = v
                beta = np.linalg.norm(omega)
                if beta > 1e-6:
                    v = omega / beta
                else:
                    break

            # Save current values
            self._update_V_matrix(V_matrix, v, it)

            # Create next basis vector
            forces = [
                self._get_forces_with_displacement(factor * displacement, v).flatten()
                for factor in factors
            ]
            omega = projection_function(*forces, displacement)

            # Save current values
            alpha = np.dot(omega, v)
            self._update_T_matrix(T_diag, T_offdiag, alpha, beta, it)

            # Orthogonaliza basis
            omega = omega - alpha * v - beta * oldv

        T_eigvals, T_eigvecs = eigh_tridiagonal(T_diag, T_offdiag)
        # To be continued

    def _get_forces_with_displacement(self, displacement, vector):
        posinp = deepcopy(self._ground_state)
        translation = (displacement * vector).reshape(-1, 3)

        # Loop on atoms is slow, but it's negligible
        # Posinp needs to be changed to make this more efficient
        for i, atom in enumerate(posinp):
            atom.position += translation[i]

        job = Job(posinp=posinp, calculator=self.calculator)
        job.run("forces")
        return job.results["forces"]

    def _update_V_matrix(self, V_matrix, v, it):
        V_matrix[:, it] = v

    def _update_T_matrix(self, T_diag, T_offdiag, alpha, beta, it):
        T_diag[it] = alpha
        if it > 0:
            T_offdiag[it - 1] = beta

    def _get_projection_function(self, num_eval):
        # Simulate the Hessian * v matricial product
        # without explicitely knowing the Hessian
        if num_eval == 3:

            def project_hessian(F1, F2, F3, displacement):
                return (6 * displacement) ** (-1) * (-18 * F1 + 9 * F2 - 2 * F3)

        elif num_eval == 4:

            def project_hessian(F1, F2, Fmoins1, Fmoins2, displacement):
                return (12 * displacement) ** (-1) * (
                    F2 - 8 * F1 + 8 * Fmoins1 - Fmoins2
                )

        else:
            raise NotImplementedError()
        return project_hessian

    def _get_factors(self, num_eval):
        if num_eval == 3:
            factors = [1, 2, 3]
        elif num_eval == 4:
            factors = [1, 2, -1, -2]
        else:
            raise NotImplementedError()
        return factors
