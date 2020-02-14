"""
TODO:
* REVERSE EL_EF (Code exists somewhere)
* SPARSE IF BELOW THRESHOLD SIZE
* HIDDEN METHODS WHICH DO LITTLE WORK, E.G. GRAD ASSUMING SYSTEM IS SOLVED
    AND VISIBLE METHODS WHICH DO MORE WORK BUT ARE SAFE
* Population.calc_ll() has a hack for model emulation (`tmp` is a float not a
    distribution as LL is estimated)
"""
import logging
import random
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import eig, norm
from scipy.sparse import eye
from scipy.sparse.linalg import bicgstab, eigs

import snepits
from snepits import _models_spec

logger = logging.getLogger(__name__)


class Model(metaclass=ABCMeta):
    """ Household model metaclass.

    Attributes:
        Ntup (tuple): Population sizes
        params (tuple): Model parameters
        seed (int): PRNG seed
        freq_dep (bool): Frequency dependent transmission if True
        get_sol (obj): Chosen solver according to `self.sparse`
        M (numpy.array): Continous-Time-Markov-Chain transition matrix
        v (numpy.array): Dominant eigenvector of system (randomly initialised)
        p (numpy.array): Solution probability vector
        grads (numpy.array): Gradient of `self.p` w.r.t. `params`

        param_l (list[str]): Name of parameters
        demog_l (list[str]): Name of risk levels
        dim (int): Number of parameters
        sparse (bool): If True use sparse solver
    """

    _eigs = False

    def __init__(self, Ntup, params, seed=None, freq_dep=True):
        """
        Args:
            Ntup (tuple): Tuple of population sizes for each class
            params (tuple): Model parameters
            seed (int, optional): PRNG seed
            freq_dep (bool, optional): Frequency dependent transmission if True
        """
        # XXX:
        self.call_counter = {"solve": 0, "grad": 0}

        if len(params) != self.dim:
            msg = f"Expected `params` of length {self.dim}, not {len(params)}"
            raise ValueError(msg)

        # Set seed
        if seed is False:
            self.seed = False
        else:
            if seed is None:
                seed = int(random.SystemRandom().randint(0, 1e6))
            logging.info(f"seed: {seed}")
            self.seed = seed
            np.random.seed(self.seed)

        if not isinstance(Ntup, np.ndarray):  # Ensure population sizes are a list
            assert 0, "Ntup is not an array"
        if len(Ntup) > 1:
            msg = f"Substructures do not total {Ntup}"
            # assert sum(Ntup)-Ntup[0] == Ntup[0], msg
        self.Ntup = Ntup
        self.N = Ntup.sum()
        self.A = len(self.demog_l)
        # Give class attributes named from demog_l with values from Ntup
        [setattr(self, self.demog_l[i], Ntup[i]) for i in range(Ntup.size)]

        # Set model parameters as tuple and individually by name
        self.params = params
        [setattr(self, self.param_l[i], params[i]) for i in range(self.dim)]

        self.freq_dep = freq_dep

        # Generate transition matrix
        self.gen_mat()
        self.v = np.random.rand(self.M.shape[0])
        self.v /= self.v.sum()
        self.z = np.zeros(self.M.shape[0])
        self.p = self.v

        # Initialise gradient variable
        self.grads = np.zeros((self.dim, self.M.shape[0]))

    def get_sol(self):
        # Solve model by sparse or dense methods
        self.call_counter["solve"] += 1
        if self.sparse is False:
            return self.__get_sol_dense()
        else:
            return self.__get_sol_sparse()

    @abstractmethod
    def gen_mat(self):
        # abstractmethod: Generate transition matrix
        pass

    def __get_sol_dense(self):
        """
        Get the stationary distribution of the model
        """
        M = self.gen_mat()  # Generate transition matrix
        try:
            e, v = eig(M)  # Eigenvalues and Eigenvectors
        except Exception as rte:
            rte.args = (rte.args[0] + " at params: %s" % list(self.params),)
            raise rte

        i = np.argmin(np.abs(e))  # Find dominant eigenvalue
        self.p = (v[:, i] / v[:, i].sum()).real  # Return dominant eigenvector
        # XXX: Hacky
        self.p += np.finfo(np.float).eps
        assert not np.any(self.p == 0)
        return self.p

    def __get_sol_sparse(self):
        self.M = self.gen_mat()  # Generate transition matrix
        self.M = (
            self.M + eye(self.M.shape[0], self.M.shape[1]) * 1e-10
        )  # Prevent singularity
        try:
            if not self._eigs:
                self.p = np.abs(bicgstab(self.M, self.z, x0=self.p)[0])
                self.p /= norm(self.p, 1)
            else:
                [e, self.v] = eigs(self.M, 1, v0=self.v, sigma=0, which="LM")
                self.p = np.abs(self.v.real.ravel() / norm(self.v.real.ravel(), 1))
        except (ValueError, RuntimeError) as rte:
            rte.args = (rte.args[0] + " at params: %s" % list(self.params),)
            logger.warning("NaNs:", np.isnan(self.p).sum())
            logger.warning("infs:", np.isinf(self.p).sum())
            raise rte

        # XXX: Hacky
        self.p += np.finfo(np.float).eps
        # assert not np.any(self.p == 0)
        return self.p

    def get_grads(self, step=1e-10):
        """
        Get gradient of log likelihood using finite differences.
        step (opt): stepsize
        """
        self.call_counter["grad"] += 1
        params = self.params
        p1 = self.p.copy()  # Store original matrix
        assert not np.any(np.isinf(np.log(p1))), (params, p1)
        tmp = np.zeros(self.dim)
        for i in range(self.dim):
            # Calculate perturbed solution:
            tmp[i - 1] = 0
            tmp[i] += step
            self.param_update(params + tmp)
            p2 = self.get_sol()
            assert not np.any(np.isinf(np.log(p2))), "p2"
            # Approximate derivate
            self.grads[i, :] = (np.log(p2) - np.log(p1)) / step

        # Return to starting state:
        self.param_update(params)
        self.p = p1
        return self.grads

    def _gen_dat(self, m):
        """
        Generate data `D` for `m` populations
        """
        self.get_sol()
        D = self.__inv_samp(m)
        return D

    def __inv_samp(self, n):
        """
        Inverse sample a probability distribution `p`, `n` times
        """
        i = 0
        p = self.p.cumsum()
        r = np.random.rand(n)
        out = np.empty_like(r, dtype=np.int)

        r.sort()
        while i < n:
            out[i] = np.searchsorted(p, r[i])
            i += 1
        return out

    def param_update(self, params):
        """
        Update current model parameters
        """
        self.params = params
        # Update each parameter by name
        [setattr(self, self.param_l[i], params[i]) for i in range(self.dim)]

    def data_transform(self, data):
        """ From state space to state index """
        from pyovpyx import el_f_gen

        return el_f_gen(self.Ntup, data)

    def idx_transform(self, idx):
        """ From state index to state space """
        from pyovpyx import idx_to_state

        return idx_to_state(idx, self.A, self.R, self.Ntup)


class Population(metaclass=ABCMeta):
    """ Population of household models metaclass

    Attributes:
        seed (int): PRNG seed
        HHsizes (numpy.array): Size of each household
        uniHHsizes (numpy.array): Unique sizes of households
        HHlookup (dict): Map from uniHHsizes to HHsizes.
           The i-th key gives the indices of HHsizes that correspond
           to the i-th element of uniHHsizes.
        params (tuple): Model parameters
        freq_dep (bool): If True, frequency dependent transmission
        m (int): Number of households in system
        N (int): Total population of system
        sub_pops (list): List of Model objects corresponding to the
            household types in the system.
        t_params (tuple): True parameters if `data` is None else None
        infected (numpy.array): Index in state-space of input data
            for each household.
        LL (float): Log-likelihood of system
        grad (numpy.array): Gradient of self.LL w.r.t. self.params
    """

    def __init__(self, sizes, params, seed=None, freq_dep=True, data=None):
        """
        Args:
            sizes (numpy.array): Array of population sizes for each class.
                One row per household, one column per class.
            params (tuple): Model parameters
            seed (int, optional): PRNG seed
            freq_dep (bool, optional): Frequency dependent transmission if True
            data (numpy.array, optional): Numbers infected for each class
                One row per household, one column per class.
        """
        # XXX:
        self.call_counter = {"LL": 0, "grad": 0}
        self.save_calc = []

        # Set seed
        if seed is None:
            seed = int(random.SystemRandom().randint(0, 1e6))
        logging.info(f"seed: {seed}")
        self.seed = seed
        np.random.seed(self.seed)

        self.HHsizes = np.array(sizes)  # Set meta-population sizes
        if self.HHsizes.ndim == 1:
            self.HHsizes = self.HHsizes.reshape((self.HHsizes.size, 1))

        # Set model parameters as tuple and individually by name
        self.params = params
        [setattr(self, self.param_l[i], params[i]) for i in range(self.dim)]
        self.freq_dep = freq_dep
        self.A = len(self.demog_l)

        self.m = self.HHsizes.shape[0]  # Number of meta-populations
        self.N = self.HHsizes.sum()  # Total population size
        self._process_HHs()  # Sort HH sizes and create lookup table
        self._gen_sub_pops()  # Initialise sub populations

        # If `data` not provided, generate data and store the true parameters
        if data is None:
            self._gen_dat()
            self.t_params = self.params
            self.data = self.idx_transform(self.infected)
        else:
            self.t_params = None
            if data.ndim == 1:
                data = data.reshape((data.size, 1))
            msg = "Num of data points does not match that of population sizes"
            assert self.m == data.shape[0], msg

            self.data = data[self.__HHidx__, :]  # Sort according to HHsizes
            self.data_orig = data.copy()

            # Transform data format into state vector index
            self.infected = np.zeros(data.shape[0], dtype=np.int)
            self.infected = self.data_transform(self.data)

    def _gen_sub_pops(self):
        # Generate an instance of class `Model` giving a sub-population for
        # each meta-population type
        self.sub_pops = []
        for i in range(self.uniHHsizes.shape[0]):
            self.sub_pops.append(
                self.subclass(self.uniHHsizes[i, :], self.params, self.freq_dep)
            )

    def _process_HHs(self):
        from numpy import (
            sort,
            lexsort,
            ascontiguousarray,
            dtype,
            void,
            unique,
            hstack,
            newaxis,
            where,
            all,
        )

        # lexsort by total sizes and then sizes starting from last column
        idx = lexsort(
            hstack([self.HHsizes.sum(axis=1)[:, newaxis], self.HHsizes]).T[::-1, :]
        )
        self.HHsizes = self.HHsizes[idx, :]

        # Find unique combinations of meta-population sizes by viewing each row
        # as one long element
        b = ascontiguousarray(self.HHsizes).view(
            dtype((void, self.HHsizes.dtype.itemsize * self.HHsizes.shape[1]))
        )
        _, idx2 = unique(b, return_index=True)
        self.uniHHsizes = self.HHsizes[sort(idx2), :]

        # Create lookup table
        d = [
            where(all(self.HHsizes == HHs, axis=1))[0]
            for i, HHs in enumerate(self.uniHHsizes)
        ]
        self.HHlookup = {i: j for i, j in enumerate(d)}
        self.rev_HHlookup = {jj: i for i, j in enumerate(d) for jj in j}

        if len(self.demog_l) == 1:
            self.uniHHsizes = self.uniHHsizes.reshape((self.uniHHsizes.size, 1))

        self.__HHidx__ = idx  # Needed to sort data (if exists)

    def data_transform(self, data):
        """ From state space to state index """
        from numpy import array, int

        for i, HH in enumerate(self.uniHHsizes):
            hhidxs = self.HHlookup[i]
            self.infected[hhidxs] = array(
                [self.sub_pops[i].data_transform(data[j]) for j in hhidxs], dtype=int
            )
            # TODO GET THI RIGHT
        return self.infected

    def idx_transform(self, idx):
        """ From state index to state space """
        from numpy import array, int, empty

        self.data = empty((self.infected.shape[0], self.A * self.R))
        for i, HH in enumerate(self.uniHHsizes):
            hhidxs = self.HHlookup[i]
            self.data[hhidxs] = array(
                [self.sub_pops[i].idx_transform(self.infected[j]) for j in hhidxs],
                dtype=int,
            )
        return self.data

    def calc_LL(self, LL_old=0, r=0):
        """ Calculate log-likelihood of meta-population model """
        self.call_counter["LL"] += 1
        # Return -inf if any parameter is not strictly positive
        if np.any([(p <= 0 or np.isinf(p)) for p in self.params]):
            self.LL = -np.inf
            self._unsolved = False
            return self.LL

        LL = 0
        for i in range(self.uniHHsizes.shape[0]):  # For each meta-pop type
            # Get data of meta-pops of this type:
            inf_ = self.infected[self.HHlookup[i]]
            uI, uIc = np.unique(inf_, return_counts=True)

            tmp = self.sub_pops[i].get_sol()  # Solve sub-population

            # XXX: self.sub_pops may be emulated not solved exactly
            if isinstance(tmp, float):
                LL += tmp
            else:
                # Add log likelihood of individual meta-population to total LL:
                for j_ind, j in enumerate(uI):
                    LL += np.log(tmp[int(j)]) * uIc[j_ind]

                # Early reject
                if (
                    not (np.log(r) < (LL - LL_old))
                    # if ((np.log(r) >= (LL - LL_old))
                    and (i != self.uniHHsizes.shape[0] - 1)
                ):
                    self.save_calc.append((r, LL, LL_old, i))
                    return -np.inf

        self.LL = LL

        if np.isnan(self.LL):
            warnings.warn(
                "NaN LL encountered due to poor conditioning,\
                          attempting reconditioning..."
            )
            for i in range(self.uniHHsizes.shape[0]):
                p = self.sub_pops[i].p
                # Find negative probabilities
                negs = np.nonzero(p < 0)[0]
                for j in negs:
                    # If small negative, make correction
                    if p[j] > -np.finfo(np.double).eps:
                        p[j] = np.finfo(np.double).eps
            LL = 0
            for i in range(self.uniHHsizes.shape[0]):  # For each meta-pop type
                # Get data of meta-pops of this type:
                inf_ = self.infected[self.HHlookup[i]]
                uI, uIc = np.unique(inf_, return_counts=True)

                # Add log likelihood of individual meta-population to total LL
                for j_ind, j in enumerate(uI):
                    LL += np.log(self.sub_pops[i].p[int(j)]) * uIc[j_ind]
            self.LL = LL

        if np.isnan(self.LL):
            print("NaN params:", list(self.params))
            for i in range(len(self.uniHHsizes)):
                if np.any(self.sub_pops[i].p < 0):
                    logger.warning(
                        "Log likelihood is NaN because a solution\
                                     is negative & likely poorly conditioned"
                    )
            logger.warning(
                "Log likelihood is NaN. No solution is negative\
                             therefore not likely a conditioning problem"
            )
            self.LL = -np.inf
        self._unsolved = False

        return self.LL

    def treat(self, efficacy=0.95, coverage=0.8):
        """ """
        for Hi in self.sub_pops:
            pass
        raise NotImplementedError()

    def calc_grad(self, step=1e-5, recalc=False):
        """
        Get gradient of log likelihood of all meta-populations using
        finite differences.

        Args:
            step (float, optional): Step size for finite differences
            recalc (bool, optional): If False assumes system already
                solved for current params
        """
        self.call_counter["grad"] += 1
        # Solve system
        if recalc or self._unsolved:
            [pop.get_sol() for pop in self.sub_pops]

        # Return -inf if any parameters are not strictly positive
        if np.any([p <= 0 for p in self.params]):
            self.LL = -np.inf
            return -np.inf

        self.grad = np.zeros(self.dim)
        LL = 0
        for i in range(self.uniHHsizes.shape[0]):  # For each meta-pop type
            # Get data of meta-pops of this type:
            inf_ = self.infected[self.HHlookup[i]]
            uI, uIc = np.unique(inf_, return_counts=True)

            # Solve sub-population and calculate gradients
            tmp = self.sub_pops[i].p.copy()
            tmp2 = self.sub_pops[i].get_grads(step)

            # Add log likelihood of individual meta-population to total LL:
            for j_ind, j in enumerate(uI.astype(int)):
                LL += np.log(tmp[j]) * uIc[j_ind]
                self.grad += tmp2[:, j] * uIc[j_ind]

        self.LL = LL
        self._unsolved = False
        return self.grad

    def _gen_dat(self):
        """ Generate data from parameters supplied at initialisation """
        data = []
        for i in range(self.uniHHsizes.shape[0]):  # For each meta-pop type
            # Get number of meta-pops of this type:
            m = self.HHlookup[i].size
            # Generate `m` data-points for this meta-pop type
            data.append(self.sub_pops[i]._gen_dat(m))
        self.infected = np.concatenate(data)  # Concatenate data to np array
        return self.infected

    def param_update(self, params):
        """ Update current model parameters

        Args:
            params (tuple): New parameters
        """
        self.params = params
        # Update each parameter by name
        [setattr(self, self.param_l[i], params[i]) for i in range(self.dim)]
        for i in self.sub_pops:  # Update parameters of each sub-population:
            i.param_update(params)
        self._unsolved = True
