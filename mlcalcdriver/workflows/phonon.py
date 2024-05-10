r"""
The :class:`Phonon` class allows to compute the normal modes
and vibration energies of a system using a machine
learning trained model.
"""

from mlcalcdriver.calculators.calculator import Calculator, DummyCalculator
from mlcalcdriver.globals import ANG_TO_B, B_TO_ANG, EV_TO_HA, HA_TO_CMM1, AMU_TO_EMU
from mlcalcdriver.workflows import Geopt
from copy import deepcopy
from mlcalcdriver import Job, Posinp
import numpy as np
import scipy
import warnings


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
        relax=True,
        finite_difference=False,
        translation_amplitudes=None,
        low_memory=False,
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
            Default is `True`.
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
        self.low_memory = low_memory

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
            If `True`, which is default, the initial positions are relaxed
            before the phonon properties are calculated. Recommended,
            especially if more than one model is used.
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

    def run(self, batch_size=128, **kwargs):
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

    def _post_proc(self, job, use_jax=False):
        r"""
        Calculates the energies and normal modes from the results
        obtained from the model.
        """
        self.dyn_mat = self._compute_dyn_mat(job)
        self.energies, self.normal_modes = self._solve_dyn_mat(use_jax=use_jax)
        self.energies *= HA_TO_CMM1

    def _compute_dyn_mat(self, job):
        r"""
        Computes the dynamical matrix
        """
        hessian = self._compute_hessian(job)
        if not self.low_memory:
            masses = self._compute_masses()
            return hessian / masses
        else:
            masses = np.array(
                [atom.mass for atom in self._ground_state for _ in range(3)],
            )
            for i in range(int(hessian.shape[0] / 3)):
                hessian[3 * i : 3 * (i + 1)] = hessian[3 * i : 3 * (i + 1)] / np.sqrt(
                    masses * masses[3 * i]
                )
            return hessian

    def _compute_masses(self):
        r"""
        Creates the masses matrix
        """
        to_mesh = np.array(
            [atom.mass for atom in self._ground_state for _ in range(3)],
        )
        m_i, m_j = np.meshgrid(to_mesh, to_mesh)
        return np.sqrt(m_i * m_j)

    def _compute_hessian(self, job):
        r"""
        Computes the hessian matrix from the forces
        """
        n_at = len(self.posinp)
        if "hessian" in job.results.keys():
            h = job.results["hessian"].reshape(3 * n_at, 3 * n_at)
            return (h + h.T) / 2.0
        else:
            warnings.warn(
                "The hessian matrix is approximated by a numerical derivative."
            )
            hessian = np.zeros((3 * n_at, 3 * n_at))
            forces = np.array(job.results["forces"])
            for i in range(3 * n_at):
                hessian[i, :] = (
                    -forces[4 * i].flatten()
                    + forces[4 * i + 3].flatten()
                    + 8 * (forces[4 * i + 1].flatten() - forces[4 * i + 2].flatten())
                ) / (12 * self.translation_amplitudes)
            return -(hessian + hessian.T) / 2.0

    def _solve_dyn_mat(self, use_jax=False):
        r"""
        Obtains the eigenvalues and eigenvectors from
        the dynamical matrix
        """
        if use_jax:
            try:
                import jax.scipy

                jax.config.update("jax_traceback_filtering", "off")
                use_jax = True
            except ModuleNotFoundError:
                print("Jax not installed, defaults to basic scipy library.")
                use_jax = False

        if use_jax:
            eigs, vecs = jax.scipy.linalg.eigh(self.dyn_mat)
        else:
            eigs, vecs = scipy.linalg.eigh(self.dyn_mat)

        eigs *= EV_TO_HA * B_TO_ANG**2 / AMU_TO_EMU
        eigs = np.sign(eigs) * np.sqrt(np.where(eigs < 0, -eigs, eigs))
        return eigs, vecs


class PhononFromHessian(Phonon):
    r"""
    Similar to the main Phonon class, but can be used when calculating
    many structures with the same hessian matrix (isotopes study). Saves
    the time to compute the hessian matrix each time
    """

    def __init__(self, posinp, hessian, sparse=False, sparse_kwargs=None):
        r"""
        Parameters
        ----------
        posinp : mlcaldriver.Posinp
            Initial positions of the system under consideration.
        hessian : np.ndarray or str
            Hessian matrix calculated before instanciating this class.
            Can be the array or a path to .npy file (created with np.save).
        """
        super().__init__(
            posinp=posinp,
            calculator=DummyCalculator(),
            relax=False,
            finite_difference=False,
            low_memory=True,
        )
        self.sparse = sparse
        if self.sparse:
            self.sparse_kwargs = sparse_kwargs
        self.hessian = hessian

    @property
    def hessian(self):
        return self._hessian

    @hessian.setter
    def hessian(self, hessian):
        if self.sparse:
            assert isinstance(hessian, scipy.sparse._compressed._cs_matrix)
            self._hessian = scipy.sparse.csr_array(hessian)
        else:
            if isinstance(hessian, str):
                hessian = np.load(hessian)

            if isinstance(hessian, np.ndarray):
                assert hessian[0].shape == (
                    3 * len(self.posinp),
                    3 * len(self.posinp),
                ), f"The hessian shape {hessian.shape} does not match the number of atoms {len(self.posinp)}"
                self._hessian = hessian
            else:
                raise TypeError("The hessian matrix should be a numpy array.")

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, sparse):
        self._sparse = bool(sparse)

    def run(self, use_jax=False, sparse_kwargs={}):
        if not self.sparse:
            job = Job(posinp=self.posinp, calculator=self.calculator)
            job.results["hessian"] = self.hessian
            self._post_proc(job, use_jax=use_jax)
        else:
            self._solve_sparse_hessian(kwargs=sparse_kwargs)

    def _solve_sparse_hessian(self, kwargs):
        self.dyn_mat = self._compute_sparse_dyn_mat()
        self.energies, self.normal_modes = self._solve_sparse_dyn_mat(kwargs=kwargs)
        self.energies *= HA_TO_CMM1

    def _compute_sparse_dyn_mat(self):
        self.hessian = (self.hessian + self.hessian.T) / 2
        masses = np.array(
            [atom.mass for atom in self.posinp for _ in range(3)],
        )
        for i in range(self.hessian.shape[0]):
            self.hessian.data[
                self.hessian.indptr[i] : self.hessian.indptr[i + 1]
            ] = self.hessian.data[
                self.hessian.indptr[i] : self.hessian.indptr[i + 1]
            ] / np.sqrt(
                masses[i]
                * masses[
                    self.hessian.indices[
                        self.hessian.indptr[i] : self.hessian.indptr[i + 1]
                    ]
                ]
            )

    def _solve_sparse_dyn_mat(self, kwargs={}):
        eigs, vecs = scipy.sparse.linalg.eigsh(
            self.hessian,
            which="LM",
            **kwargs,
        )
        eigs *= EV_TO_HA * B_TO_ANG**2 / AMU_TO_EMU
        eigs = np.sign(eigs) * np.sqrt(np.abs(eigs))
        return eigs, vecs
