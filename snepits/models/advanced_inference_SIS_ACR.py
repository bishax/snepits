import logging
import os
import time

import numpy as np
import pymc3 as pm
import scipy.stats as ss
import theano.tensor as tt

import snepits
from snepits.models.advanced_inference import run_experiment
from snepits.models.models_spec import (SIS_ACR_orthogonal_reparam,
                                        SIS_ACR_pop_gen)
from snepits.models.pymc_wrapper import PymcWrapper

logger = logging.getLogger(__name__)


def get_model():
    # Model setup
    m = SIS_ACR_pop_gen

    sizes = np.ones((100, 3), dtype=np.int) * 1
    sizes[:, 0] = sizes[:, 1:].sum(1)
    params = np.array([0.5, 1, 2, 5, 1, 1, 0.01])
    model = m(
        SIS_ACR_orthogonal_reparam, sizes=sizes, params=params, data=None
    )

    priors = {k: ss.halfnorm(0, 5) for k in model.param_l}

    return model, priors


def pymc_model(model):
    with pm.Model() as pm_model:
        beta_c = pm.HalfNormal('beta_c', sd=5)
        beta_l = pm.HalfNormal('beta_l', sd=5)
        beta_h = pm.HalfNormal('beta_h', sd=5)
        rho_c = pm.HalfNormal('rho_c', sd=5)
        rho_h = pm.HalfNormal('rho_h', sd=5)
        g = pm.HalfNormal('g', sd=5)
        eps = pm.HalfNormal('eps', sd=5)
        theta = tt.as_tensor_variable([beta_c, beta_l, beta_h, rho_c, rho_h, g, eps])
        loglike = PymcWrapper(model)
        pm.DensityDist("likelihood", lambda v: loglike(v), observed={"v": theta})
    return pm_model


if __name__ == "__main__":
    chains = 4
    tune, draws = 1000, 2000
    tune_opt, draws_opt = 1000, 20000
    tune_adapt, draws_adapt = 25000, 50000
#    tune, draws = 10, 20
#    tune_opt, draws_opt = 10, 200
#    tune_adapt, draws_adapt = 250, 500
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