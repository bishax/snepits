import logging
import os
import time

import numpy as np
import pymc3 as pm
import scipy.stats as ss
import theano.tensor as tt

import snepits
from snepits.models.advanced_inference import run_experiment
from snepits.models.models_spec import SIS_AC_pop
from snepits.models.pymc_wrapper import PymcWrapper

logger = logging.getLogger(__name__)


def get_model():
    # Model setup
    m = SIS_AC_pop
    sizes = np.ones((100, 3), dtype=np.int) * 2
    sizes[:, 0] = sizes[:, 1:].sum(1)
    params = np.array([1, 0.8, 1.5, 0.4])
    model = m(sizes=sizes, params=params, data=None)

    priors = {k: ss.halfnorm(0, 5) for k in model.param_l}

    return model, priors


def pymc_model(model):
    with pm.Model() as pm_model:
        beta_A = pm.HalfNormal("beta_A", sd=2)
        beta_C = pm.HalfNormal("beta_C", sd=2)
        rho_C = pm.HalfNormal("rho_C", sd=2)
        eps = pm.HalfNormal("eps", sd=2)
        theta = tt.as_tensor_variable([beta_A, beta_C, rho_C, eps])
        loglike = PymcWrapper(model)
        pm.DensityDist("likelihood", lambda v: loglike(v), observed={"v": theta})
    return pm_model


if __name__ == "__main__":
    chains = 4
    tune, draws = 1000, 2000
    tune_opt, draws_opt = 1000, 20000
    tune_adapt, draws_adapt = 25000, 50000
    model, priors = get_model()
    pm_model = pymc_model(model)

    k_l = ['NUTS_pymc', 'MH_pymc', 'MH', 'MH_opt', 'MALA', 'MALA_opt']

    samplers, trace = run_experiment(
        model,
        priors,
        pm_model,
        chains,
        tune,
        draws,
        tune_opt,
        draws_opt,
        tune_adapt,
        draws_adapt,
        k_l,
    )
