import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import pairplot

import pymc3 as pm
from pymc3.stats import autocov


def _get_neff(x):
    """Compute the effective sample size for a 2D array
    """
    trace_value = x.T
    nchain, n_samples = trace_value.shape

    acov = np.asarray([autocov(trace_value[chain]) for chain in range(nchain)])

    chain_mean = trace_value.mean(axis=1)
    chain_var = acov[:, 0] * n_samples / (n_samples - 1.0)
    acov_t = acov[:, 1] * n_samples / (n_samples - 1.0)
    mean_var = np.mean(chain_var)
    var_plus = mean_var * (n_samples - 1.0) / n_samples
    var_plus += np.var(chain_mean, ddof=1)

    rho_hat_t = np.zeros(n_samples)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov_t)) / var_plus
    rho_hat_t[1] = rho_hat_odd
    # Geyer's initial positive sequence
    max_t = 1
    t = 1
    while t < (n_samples - 2) and (rho_hat_even + rho_hat_odd) >= 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        max_t = t + 2
        t += 2

    # Geyer's initial monotone sequence
    t = 3
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2
    ess = nchain * n_samples
    ess = ess / (-1.0 + 2.0 * np.sum(rho_hat_t))
    return ess


def generate_neff(trace_values):
    """ Calculate effective number of samples

    Args:

    trace_values (list): List of 1D numpy arrays.
        Each list element corresonds to the trace of one chain.
    """
    x = np.array(trace_values)
    shape = x.shape

    # Make sure to handle scalars correctly, adding extra dimensions if
    # needed. We could use np.squeeze here, but we don't want to squeeze
    # out dummy dimensions that a user inputs.
    if len(shape) == 2:
        x = np.atleast_3d(trace_values)

    # Transpose all dimensions, which makes the loop below
    # easier by moving the axes of the variable to the front instead
    # of the chain and sample axes.
    x = x.transpose()

    # Get an array the same shape as the var
    _n_eff = np.zeros(x.shape[:-2])

    # Iterate over tuples of indices of the shape of var
    for tup in np.ndindex(*list(x.shape[:-2])):
        _n_eff[tup] = _get_neff(x[tup])

    if len(shape) == 2:
        return _n_eff[0]

    return np.transpose(_n_eff)


def trace_pairplot(trace_l, labels=None, t_params=None, diag_kind="auto", **kwargs):
    """

    Args:
        trace_l (list): List of pymc3 traces
        labels (list, optional): List of trace names
        t_params (list, optional): True parameters
        diag_kind ({'auto', 'hist', 'kde'}, optional):
        **kwargs (dict, optional): Kwargs passed to seaborn.pairplot

    Returns:
        seaborn.axisgrid.PairGrid
    """

    if len(trace_l) > 1:
        df_trace = []
        for i, trace in enumerate(trace_l):
            if labels is None:
                nm = i
            else:
                nm = labels[i]
            if not isinstance(trace, pd.DataFrame):
                trace = pm.trace_to_dataframe(trace)
            trace = trace.assign(method=nm)
            df_trace.append(trace)
        df_trace = pd.concat(df_trace)
        hue = "method"
        vars = df_trace.columns.drop("method")
    else:
        if not isinstance(trace_l[0], pd.DataFrame):
            df_trace = pm.trace_to_dataframe(trace_l[0])
        else:
            df_trace = trace_l[0]
        hue = None
        vars = df_trace.columns

    # ax = pairplot(df_trace, diag_kind='kde', hue=hue, **kwargs)
    ax = sns.PairGrid(df_trace, hue=hue, diag_sharey=False, vars=vars)
    if ((hue is None) and (diag_kind != "kde")) or (diag_kind == "hist"):
        ax.map_diag(plt.hist, alpha=0.8, density=True)
    else:
        ax.map_diag(sns.kdeplot)
    ax.map_lower(plt.scatter, **kwargs)

    if t_params is not None:
        c = "r"
        D = len(t_params)
        for i in range(D):
            ax.axes[i][i].axvline(x=t_params[i], c=c)
            for j in range(i + 1, D):
                ax.axes[j][i].plot(t_params[i], t_params[j], "ro")
                ax.axes[i][j].axis("off")

    a = ax.axes[0][-1]
    fontsize = mpl.rcParams["axes.labelsize"]
    title = ax._hue_var
    handles = [
        ax._legend_data.get(hue, mpl.patches.Patch(alpha=0, linewidth=0))
        for hue in ax.hue_names
    ]
    leg = a.legend(handles, ax.hue_names, loc="center", fontsize=fontsize)
    leg.set_title(title, prop={"size": fontsize})

    return ax
