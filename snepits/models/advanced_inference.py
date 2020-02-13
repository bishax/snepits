import json
import logging
import os
import time

import numpy as np
import pandas as pd
import pymc3 as pm
from joblib import Parallel, delayed, dump

import snepits
from snepits.models.mcmc_samplers import AMALA, AdMetHas
from snepits.utils.pymc3 import generate_neff
from snepits.utils.utils import timer

logger = logging.getLogger(__name__)


def run_experiment(
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
    k_l=None,
):

    if k_l is None:
        k_l = ["NUTS_pymc", "MH_pymc", "MH", "MH_opt", "MALA", "MALA_opt"]
    elif "NUTS_pymc" not in k_l:
        k_l += "NUTS_pymc"
    else:
        pass

    def save_times():
        with open(f"{base_dir}/times.json", "w") as f:
            logger.info(f"Saving times: {dict(optimal=t_opt, adapt=t)}")
            json.dump(dict(optimal=t_opt, adapt=t), f)

    start_time = time.strftime("%H%M%d%m")
    base_dir = (
        f"{snepits.project_dir}/models/n_eff_{model.subclass.__name__}_{start_time}"
    )
    try:
        os.makedirs(base_dir, exist_ok=False)
        logger.info(f"Making directory: {base_dir}")
    except FileExistsError as e:
        logger.info(e)
        base_dir += "_2"
        os.makedirs(base_dir, exist_ok=False)
        logger.info(f"Making directory: {base_dir}")

    dump(model, f"{base_dir}/{model.subclass.__name__}_model.pkl")

    samplers, t, trace, t_opt, trace_opt = {}, {}, {}, {}, {}

    def gen_fname(k):
        fname = f"{base_dir}/{k}_{sampler.__name__}_{model.subclass.__name__}"
        return fname

    ## NUTS
    k = "NUTS_pymc"
    pm_sample_kwargs = dict(
        draws=draws, tune=tune, discard_tuned_samples=True, chains=chains, cores=1,
    )
    t[k], trace_HMC = run_pymc(pm_model, pm_sample_kwargs)
    pm.save_trace(trace_HMC, f"{base_dir}/{k}_{model.subclass.__name__}_trace")
    trace[k] = [trace_HMC.get_values(var, combine=False) for var in model.param_l]
    trace_opt[k] = trace[k]

    t_opt[k] = t[k] * draws / (tune + draws)
    trace_opt[k] = trace[k]
    save_times()

    # Empirical covariance for "optimal" sampling
    if chains > 1:
        cov_hat = np.mean(
            np.array([np.cov(np.array(trace[k])[:, i, :]) for i in range(chains)]),
            axis=0,
        )
    else:
        cov_hat = np.cov(np.array(trace[k]))
    np.save(f"{base_dir}/{k}_empirical_covariance", cov_hat)

    ## MH pymc
    k = "MH_pymc"
    if k in k_l:
        pm_sample_kwargs = dict(
            step=pm.Metropolis(model=pm_model),
            draws=draws_adapt,
            tune=tune_adapt,
            discard_tuned_samples=True,
            chains=chains,
            cores=chains,
        )
        t[k], trace_MH = run_pymc(pm_model, pm_sample_kwargs)
        pm.save_trace(trace_MH, f"{base_dir}/{k}_{model.subclass.__name__}_trace")
        trace[k] = [trace_MH.get_values(var, combine=False) for var in model.param_l]
        trace_opt[k] = trace[k]

        t_opt[k] = t[k] * draws_adapt / (tune_adapt + draws_adapt)
        trace_opt[k] = trace[k]
        save_times()

    ## MH
    sampler = AdMetHas
    t0 = 1000
    # Optimal
    k = "MH_opt"
    if k in k_l:
        fname = gen_fname(k)
        sample_kwargs = dict(
            model=model,
            t0=t0,
            niters=draws_opt + tune_opt,
            b=tune_opt,
            priors=priors,
            fname=fname,
            cov=cov_hat,
            _adapt_cov=False,
            init=model.t_params,
        )
        t_opt[k], (samplers[k], trace_opt[k]) = run_sampler(
            sampler, chains, sample_kwargs
        )
        t_opt[k] *= draws_opt / (draws_opt + tune_opt)
        save_times()

    # Adapt
    k = "MH"
    if k in k_l:
        fname = gen_fname(k)
        sample_kwargs = dict(
            model=model,
            t0=t0,
            niters=draws_adapt + tune_adapt,
            b=tune_adapt,
            priors=priors,
            fname=fname,
            init="rand",
        )
        t[k], (samplers[k], trace[k]) = run_sampler(sampler, chains, sample_kwargs)
        save_times()

    ## MALA
    sampler = AMALA
    t0 = 0
    # Optimal
    k = "MALA_opt"
    if k in k_l:
        fname = gen_fname(k)
        sample_kwargs = dict(
            model=model,
            t0=t0,
            niters=draws_opt + tune_opt,
            b=tune_opt,
            priors=priors,
            fname=fname,
            init=model.t_params,
            cov=cov_hat,
            _adapt_cov=False,
        )
        t_opt[k], (samplers[k], trace_opt[k]) = run_sampler(
            sampler, chains, sample_kwargs
        )
        t_opt[k] *= draws_opt / (draws_opt + tune_opt)
        save_times()

    # Adapt
    k = "MALA"
    t0 = 1000
    if k in k_l:
        fname = gen_fname(k)
        sample_kwargs = dict(
            model=model,
            t0=t0,
            niters=draws_adapt + tune_adapt,
            b=tune_adapt,
            priors=priors,
            fname=fname,
            init="rand",
            delta=1,
        )
        t[k], (samplers[k], trace[k]) = run_sampler(sampler, chains, sample_kwargs)
        save_times()

    if chains > 1:
        n_eff_s = effective_samples_second(trace, t, var_names=model.param_l)
        n_eff_s.to_hdf(f"{base_dir}/n_eff_s.hdf", key="adapt")
        n_eff_s_opt = effective_samples_second(
            trace_opt, t_opt, var_names=model.param_l
        )
        n_eff_s_opt.to_hdf(f"{base_dir}/n_eff_s.hdf", key="opt")
        logger.info(n_eff_s)
        logger.info(n_eff_s_opt)
    else:
        logger.error("No effective samples calculated")

    return samplers, trace


@timer
def run_pymc(pm_model, sample_kwargs={}):
    with pm_model:
        trace = pm.sample(**sample_kwargs)
    return trace


@timer
def run_sampler(sampler, chains, sample_kwargs):
    def run_chain(chain_idx, sample_kwargs):
        sample_kwargs["chain_idx"] = chain_idx
        s = sampler(**sample_kwargs)
        s.run()
        s.save()
        return s

    s_l = Parallel(n_jobs=chains, backend="multiprocessing")(
        delayed(run_chain)(chain, sample_kwargs) for chain in range(chains)
    )

    param_l = s_l[0].model.param_l
    return (
        s_l,
        [
            [s.params[s.b :, i] for chain, s in enumerate(s_l)]
            for i, _ in enumerate(param_l)
        ],
    )


def _per_second(t, n_eff):
    return {k: v / t for k, v in n_eff.items()}


def effective_samples_second(trace, t, var_names=None, k_l=None):
    """ Get the effective samples per second (ESS/s) of MCMC samplers

    Args:
        trace (dict): Keys correspond to different samplers.
            Values are array-like with dimensions (parameters, chains, samples)
        t (dict): Keys correspond to different samplers, values are time taken.
        k_l (list-like, optional): Keys in `t`, `trace` to calculate ESS/s for

    Returns:
        pandas.DataFrame
    """
    if k_l is None:
        k_l = list(set(t.keys()) & set(trace.keys()))
    if var_names is None:
        raise NotImplementedError()

    n_eff = {k: dict() for k in k_l}
    for i, var in enumerate(var_names):
        for k in k_l:
            n_eff[k][var] = generate_neff(trace[k][i])

    n_eff_s = pd.DataFrame({k: _per_second(t[k], n_eff[k]) for k in n_eff.keys()})

    return n_eff_s
