"""
from models_spec import SIS, SIS_pop

# Single household
model = SIS(10, (2, 0.5), freq_dep=True)

# Multiple households - no data
model = SIS_pop([10, 10], (2, 0.5), freq_dep=True, data=None)

# Multiple households - data
model = SIS_pop([10, 10], (2, 0.5), freq_dep=True, data=[4, 5])
"""
import logging

from numpy import arange, diag

import snepits
from snepits.models.models_meta import Model, Population

logger = logging.getLogger(__name__)


class SIS(Model):
    """ SIS model class.

    Attributes:
        param_l (list[str]): Name of parameters
        demog_l (list[str]): Name of risk levels
        dim (int): Number of parameters
        sparse (bool): If True use sparse solver


    Args:
        N (int): Population size
        params (tuple): Model parameters (beta, eps)
            beta - Within population transmission rate
            eps - External Force of infection
        freq_dep (bool, optional): Frequency dependent transmission if True
    """

    param_l = ["beta", "eps", "alpha"]
    demog_l = ["N"]
    R = 1
    dim = len(param_l)
    sparse = False

    def gen_mat(self):
        """
        Generate transition matrix `M` of model
        """
        N = self.N
        beta = self.beta
        alpha = self.alpha
        if self.freq_dep:
            beta *= 1 / (N - 1) ** alpha
        eps = self.eps

        x = arange(N + 1)
        self.M = diag(x[1:], k=1) + diag((N - x[:-1]) * (beta * x[:-1] + eps), k=-1)
        self.M[x, x] = -self.M.sum(axis=0)
        return self.M

    def __str__(self):
        return "SIS: N = %d" % (self.N)


class SIS_pop(Population):
    """ SIS_pop model class.

    Attributes:
        subclass (obj): Class of sub_pops
        param_l (list[str]): Name of parameters
        demog_l (list[str]): Name of risk classes
        dim (int): Number of parameters

    Args:
        sizes (numpy.array): Array of population sizes for each class.
            One row per household, one column per class.
        params (tuple): Model parameters (beta, eps)
            beta - Within population transmission rate
            eps - External Force of infection
        freq_dep (bool, optional): Frequency dependent transmission if True
        data (numpy.array, optional): Array of numbers infected for each class
            One row per household, one column per class.
    """

    subclass = SIS
    param_l = subclass.param_l
    demog_l = subclass.demog_l
    dim = len(param_l)
    R = subclass.R

    def data_transform(self, data):
        return data

    def __str__(self):
        return "SIS_pop: m = %d, n = %d" % (self.m, self.N)
