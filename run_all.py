"""
End-to-end, reproducible pipeline for the four-model volatility comparison.

Produces the numbers behind every table in the thesis from one command:
    in-sample estimates (+ std errors), full-sample QLIKE/DM/MCS,
    full-sample VaR backtests, and the regime-conditional QLIKE/DM/MCS/VaR.

Key correctness points (these differ from the old run_experiment.py):
  * QLIKE is evaluated on the CONDITIONAL VARIANCE for every model
    (the score-driven scale is multiplied by nu/(nu-2)).
  * VaR uses the correct quantile rescaling: Gaussian -> Phi^{-1};
    GARCH-t -> sqrt((nu-2)/nu)*t^{-1} (unit-variance std t);
    t-GAS / Beta-t-EGARCH -> t^{-1} on the scale.
  * MCS p-values are realigned to the correct model (see evaluation.mcs).

    python run_all.py --csv sp500_returns.csv --out output --window 1500
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from models import MODELS
from evaluation import (qlike, diebold_mariano, mcs,
                        kupiec, christoffersen, dq_engle_manganelli)

LEVELS = (0.01, 0.05)
REFIT = {"GARCH-G": 5, "GARCH-t": 5, "t-GAS": 20, "Beta-t-EGARCH": 20}


def load_returns(csv):
    """Return (dates, r) with r = 100*log-return, robust to column naming."""
    df = pd.read_csv(csv)
    cols = {c.lower(): c for c in df.columns}
    date_col = next((cols[k] for k in ("date", "datetime") if k in cols), df.columns[0])
    dates = pd.to_datetime(df[date_col]).to_numpy()
    if "logreturn" in cols:
        r = df[cols["logreturn"]].to_numpy(float)
        if np.nanmax(np.abs(r)) < 1.0:          # convert to percentage points
            r = 100.0 * r
    elif "r" in cols:
        r = df[cols["r"]].to_numpy(float)
    else:
        price = next((cols[k] for k in ("adjclose", "adj close", "close", "price") if k in cols), None)
        if price is None:
            raise ValueError(f"No return or price column in {list(df.columns)}")
        p = df[price].to_numpy(float); r = 100.0 * np.diff(np.log(p)); dates = dates[1:]
    m = ~np.isnan(r)
    return dates[m], r[m]


def cond_variance(s2_next, kw):
    if kw.get("dist") == "t_scale":
        nu = kw["nu"]; return s2_next * nu / (nu - 2)
    return s2_next


def var_levels(s2_next, kw):
    sig = np.sqrt(s2_next); out = {}
    if not kw:
        for p in LEVELS: out[p] = -stats.norm.ppf(p) * sig
    elif kw["dist"] == "t":
        nu = kw["nu"]
        for p in LEVELS: out[p] = -np.sqrt((nu - 2) / nu) * stats.t.ppf(p, nu) * sig
    else:
        nu = kw["nu"]
        for p in LEVELS: out[p] = -stats.t.ppf(p, nu) * sig
    return out


def std_errors(model_cls, params, y):
    p = np.asarray(params, float); k = len(p); h = 1e-4
    f = lambda q: model_cls._negloglik(q, y)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h; ej = np.zeros(k); ej[j] = h
            H[i, j] = H[j, i] = (f(p+ei+ej) - f(p+ei-ej) - f(p-ei+ej) + f(p-ei-ej)) / (4 * h * h)
    try:
        return np.sqrt(np.abs(np.diag(np.linalg.inv(H))))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


def insample_table(r, frac=0.60, n_restarts=5):
    n_in = int(frac * len(r)); y = r[:n_in]; rows = {}
    for cls in MODELS:
        fit = cls.fit(y, n_restarts=n_restarts)
        se = std_errors(cls, list(fit["params"].values()), y)
        rows[cls.name] = {"params": fit["params"], "se": dict(zip(cls.param_names, se)),
                          "loglik": fit["loglik"], "AIC": fit["AIC"], "BIC": fit["BIC"]}
    return rows


def rolling(r, cls, window, refit_every, n_restarts=3):
    T = len(r); n = T - window
    out = {k: np.full(n, np.nan) for k in ("scale", "var", "var01", "var05")}
    params = None
    for k in range(n):
        if k % refit_every == 0 or params is None:
            params = list(cls.fit(r[k:k + window], n_restarts=n_restarts)["params"].values())
        s2_next, kw = cls.forecast_next(params, r[k:k + window])
        out["scale"][k] = s2_next
        out["var"][k] = cond_variance(s2_next, kw)
        v = var_levels(s2_next, kw); out["var01"][k] = v[0.01]; out["var05"][k] = v[0.05]
    return out


def classify(r, n_in, window=22, pct=70.0):
    roll = pd.Series(r).rolling(window).std().to_numpy()
    thr = np.nanpercentile(roll[:n_in], pct)
    return roll > thr, thr


def evaluate(names, losses, r_oos, var01, var05, idx=None):
    if idx is None: idx = np.ones(len(r_oos), bool)
    bench = "GARCH-G"; out = {}
    for nm in names:
        l = losses[nm][idx]
        dm = diebold_mariano(losses[bench][idx], l) if nm != bench else {"DM": np.nan, "pvalue": np.nan}
        h01 = (r_oos[idx] < -var01[nm][idx]).astype(int)
        h05 = (r_oos[idx] < -var05[nm][idx]).astype(int)
        out[nm] = {"QLIKE": float(l.mean()), "DM": dm["DM"], "DM_p": dm["pvalue"],
                   "rate01": float(h01.mean()), "rate05": float(h05.mean()),
                   "UC01": kupiec(h01, 0.01)["pvalue"], "CC01": christoffersen(h01, 0.01)["pvalue"],
                   "DQ01": dq_engle_manganelli(h01, var01[nm][idx], 0.01)["pvalue"],
                   "UC05": kupiec(h05, 0.05)["pvalue"], "CC05": christoffersen(h05, 0.05)["pvalue"],
                   "DQ05": dq_engle_manganelli(h05, var05[nm][idx], 0.05)["pvalue"]}
    L = np.column_stack([losses[nm][idx] for nm in names])
    try:
        out["_mcs"] = mcs(L, list(names))["pvalues"]
    except Exception as e:
        print("MCS skipped:", e); out["_mcs"] = {nm: np.nan for nm in names}
    return out


def run(csv, out, window=1500, insample_frac=0.60, n_restarts=3):
    Path(out).mkdir(parents=True, exist_ok=True)
    dates, r = load_returns(csv); n_in = int(insample_frac * len(r))
    print(f"n={len(r)}  in-sample(60%)={n_in}  window={window}  oos={len(r)-window}")

    ins = insample_table(r, insample_frac, n_restarts=5)
    json.dump(ins, open(Path(out) / "insample.json", "w"), indent=2, default=float)

    names = [c.name for c in MODELS]
    fc = {c.name: rolling(r, c, window, REFIT[c.name], n_restarts) for c in MODELS}
    r_oos = r[window:]; rt2 = r_oos ** 2
    losses = {nm: qlike(rt2, fc[nm]["var"]) for nm in names}
    v01 = {nm: fc[nm]["var01"] for nm in names}; v05 = {nm: fc[nm]["var05"] for nm in names}

    turb = classify(r, n_in)[0][window:]
    res = {"oos_dates": [str(np.datetime_as_string(d, unit="D")) for d in dates[window:]],
           "n_calm": int((~turb).sum()), "n_turb": int(turb.sum()),
           "full": evaluate(names, losses, r_oos, v01, v05),
           "calm": evaluate(names, losses, r_oos, v01, v05, idx=~turb),
           "turbulent": evaluate(names, losses, r_oos, v01, v05, idx=turb)}
    del res["oos_dates"]
    json.dump(res, open(Path(out) / "results.json", "w"), indent=2, default=float)
    print("Full QLIKE:", {k: round(res['full'][k]['QLIKE'], 4) for k in names})
    print("Full MCS  :", {k: round(res['full']['_mcs'][k], 3) for k in names})
    print("Turb MCS  :", {k: round(res['turbulent']['_mcs'][k], 3) for k in names})
    return ins, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--window", type=int, default=1500)
    run(**{k: v for k, v in vars(ap.parse_args()).items()})
