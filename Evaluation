"""
Forecast-evaluation toolkit: QLIKE, Diebold-Mariano, Model Confidence Set,
and the three VaR backtests (Kupiec, Christoffersen, dynamic-quantile).
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def qlike(rt2, sigma2_hat):
    """Patton (2011) QLIKE loss, evaluated on the conditional VARIANCE."""
    sigma2_hat = np.maximum(sigma2_hat, 1e-12)
    return np.log(sigma2_hat) + rt2 / sigma2_hat


def _newey_west_var(d, h):
    n = len(d); dd = d - d.mean(); s = (dd @ dd) / n
    for l in range(1, h + 1):
        s += 2 * (1 - l / (h + 1)) * ((dd[l:] @ dd[:-l]) / n)
    return s


def diebold_mariano(loss_a, loss_b, h=1):
    """DM test of equal predictive accuracy. Positive stat -> loss_b is better."""
    d = np.asarray(loss_a) - np.asarray(loss_b); n = len(d)
    lr = _newey_west_var(d, h=max(h - 1, int(np.floor(n ** (1 / 3)))))
    if lr <= 0:
        return {"DM": np.nan, "pvalue": np.nan, "n": n}
    dm = d.mean() / np.sqrt(lr / n)
    return {"DM": float(dm), "pvalue": float(2 * (1 - stats.norm.cdf(abs(dm)))), "n": n}


def mcs(loss_matrix, model_names, alpha=0.10, block_size=10, reps=2000, method="R", seed=1):
    """
    Hansen-Lunde-Nason (2011) MCS via the `arch` package.

    NOTE: arch returns m.pvalues indexed by each model's ORIGINAL column
    position but ordered by elimination (ascending p-value). We realign each
    p-value to its own model. Zipping positionally (the previous behaviour)
    silently inverts the table.
    """
    from arch.bootstrap import MCS
    L = np.asarray(loss_matrix)
    m = MCS(L, size=alpha, reps=reps, block_size=block_size, method=method, seed=seed)
    m.compute()
    pv = m.pvalues
    pvals = np.full(len(model_names), np.nan)
    for pos, p in zip(pv.index, np.asarray(pv).flatten()):
        pvals[int(pos)] = float(p)
    included = {int(x) for x in m.included}
    in_set = np.array([i in included for i in range(len(model_names))])
    return {"pvalues": dict(zip(model_names, pvals)), "in_set": dict(zip(model_names, in_set))}


def kupiec(hits, p):
    hits = np.asarray(hits).astype(int); n = len(hits); x = int(hits.sum())
    p_hat = x / n if n > 0 else np.nan
    if x == 0 or x == n:
        return {"LR_uc": np.nan, "pvalue": np.nan, "n": n, "x": x, "p_hat": p_hat}
    lr = -2 * (x * np.log(p) + (n - x) * np.log(1 - p) - x * np.log(p_hat) - (n - x) * np.log(1 - p_hat))
    return {"LR_uc": float(lr), "pvalue": float(1 - stats.chi2.cdf(lr, 1)), "n": n, "x": x, "p_hat": p_hat}


def christoffersen(hits, p):
    hits = np.asarray(hits).astype(int); n = len(hits); x = int(hits.sum())
    if n < 3 or x == 0 or x == n:
        return {"LR_cc": np.nan, "pvalue": np.nan}
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        a, b = hits[i - 1], hits[i]
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    n0_ = n00 + n01; n1_ = n10 + n11
    if n0_ == 0 or n1_ == 0:
        return {"LR_cc": np.nan, "pvalue": np.nan}
    pi0 = n01 / n0_; pi1 = n11 / n1_; pi = (n01 + n11) / (n0_ + n1_)
    if pi in (0, 1) or pi0 in (0, 1) or pi1 in (0, 1):
        uc = kupiec(hits, p); uc.update({"LR_cc": np.nan, "pvalue": np.nan}); return uc
    ll_ind = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_dep = n00 * np.log(1 - pi0) + n01 * np.log(pi0) + n10 * np.log(1 - pi1) + n11 * np.log(pi1)
    lr_ind = -2 * (ll_ind - ll_dep)
    uc = kupiec(hits, p); lr_cc = uc["LR_uc"] + lr_ind
    return {"LR_cc": float(lr_cc), "pvalue": float(1 - stats.chi2.cdf(lr_cc, 2)),
            "LR_uc": uc["LR_uc"], "LR_ind": float(lr_ind)}


def dq_engle_manganelli(hits, var_series, p, n_lags=4):
    hits = np.asarray(hits).astype(float); var_series = np.asarray(var_series).astype(float); n = len(hits)
    if n <= n_lags + 3:
        return {"DQ": np.nan, "pvalue": np.nan}
    y = hits - p
    cols = [np.ones(n - n_lags), var_series[n_lags:]]
    for k in range(1, n_lags + 1):
        cols.append(hits[n_lags - k: n - k])
    X = np.column_stack(cols); y = y[n_lags:]
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < X.shape[1]:
        return {"DQ": np.nan, "pvalue": np.nan}
    beta = np.linalg.solve(XtX, X.T @ y)
    cov = p * (1 - p) * np.linalg.inv(XtX)
    wald = beta @ np.linalg.solve(cov, beta)
    return {"DQ": float(wald), "pvalue": float(1 - stats.chi2.cdf(wald, X.shape[1])), "df": X.shape[1]}
