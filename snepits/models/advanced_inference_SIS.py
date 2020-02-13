import logging
import os
import time

import numpy as np
import pymc3 as pm
import scipy.stats as ss
import theano.tensor as tt

import snepits
from snepits.models.advanced_inference import run_experiment
from snepits.models.models_spec import SIS_pop
from snepits.models.pymc_wrapper import PymcWrapper

logger = logging.getLogger(__name__)


def get_model():
    # Model setup
    m = SIS_pop
    sizes = np.ones(100, dtype=np.int) * 4
    params = np.array([1, 1])
    model = m(sizes=sizes, params=params, data=None)

    priors = {k: ss.uniform(0, 10) for k in model.param_l}

    return model, priors


def pymc_model(model):
    with pm.Model() as pm_model:
        beta = pm.Uniform("beta", lower=0.0, upper=10.0)
        eps = pm.Uniform("eps", lower=0.0, upper=10.0)
        theta = tt.as_tensor_variable([beta, eps])
        loglike = PymcWrapper(model)
        pm.DensityDist("likelihood", lambda v: loglike(v), observed={"v": theta})
    return pm_model


if __name__ == "__main__":
    chains = 4
    tune, draws = 1000, 2000
    tune_opt, draws_opt = 1000, 2000
    tune_adapt, draws_adapt = 25000, 50000
    model, priors = get_model()
    pm_model = pymc_model(model)

    k_l = None

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
