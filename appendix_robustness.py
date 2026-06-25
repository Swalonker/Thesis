"""
Appendix robustness checks. Fills the three appendix tables:

  tab_robust_window.tex : estimation window 1,000 / 1,500 / 2,000 trading days
  tab_robust_regime.tex : alternative regime classifiers (rolling-SD, NBER, VIX)
  tab_robust_refit.tex  : weekly vs daily refit for the two score-driven models

Run after fetch_data.py:
  python appendix_robustness.py --csv sp500_returns.csv --out output [--vix vix.csv]

NBER recession dates are hard-coded below. VIX is fetched from Yahoo (^VIX) unless
you pass --vix with a CSV of columns Date,Close.

WARNING: this is heavy. The window check is three full rolling runs and the refit
check re-estimates the score-driven models every day over ~4,788 origins. Expect it
to run for a while; run it once and cache the tables.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from models import MODELS
from evaluation import qlike, diebold_mariano, mcs
from run_all import load_returns, rolling, classify, REFIT

NAMES = [c.name for c in MODELS]
TEX = {"GARCH-G": "GARCH-Gauss", "GARCH-t": "GARCH-$t$", "t-GAS": "t-GAS", "Beta-t-EGARCH": "Beta-$t$-EGARCH"}

# US NBER recession periods intersecting the 2000-2024 sample
NBER = [("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]


def _f(x, d=4):
    try:
        return f"{x:.{d}f}"
    except (TypeError, ValueError):
        return "--"


def forecasts(r, window, refit=None, n_restarts=3):
    out = {}
    for c in MODELS:
        every = (refit or {}).get(c.name, REFIT[c.name])
        out[c.name] = rolling(r, c, window, every, n_restarts=n_restarts)
    return out


def qlike_losses(fc, r, window):
    r_oos = r[window:]; rt2 = r_oos ** 2
    return {nm: qlike(rt2, fc[nm]["var"]) for nm in NAMES}


# ---------------- Check 1: estimation-window length ----------------
def check_window(r, out, windows=(1000, 1500, 2000)):
    common = max(windows)                         # evaluate every window on r[common:]
    res = {}
    for w in windows:
        print(f"  window {w} ...", flush=True)
        lo = qlike_losses(forecasts(r, w), r, w)
        off = common - w                          # align to the common OOS start
        lo = {nm: lo[nm][off:] for nm in NAMES}
        bench = lo["GARCH-G"]
        res[w] = {nm: {"QLIKE": float(lo[nm].mean()),
                       "DM": np.nan if nm == "GARCH-G" else diebold_mariano(bench, lo[nm])["DM"]}
                  for nm in NAMES}
    lines = [r"\begin{tabular}{lcccccc}", r"\hline",
             r" & \multicolumn{3}{c}{QLIKE} & \multicolumn{3}{c}{DM vs.\ GARCH-G} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Window length & $1000$ & $1500$ & $2000$ & $1000$ & $1500$ & $2000$ \\", r"\hline"]
    for nm in NAMES:
        q = " & ".join(f"${_f(res[w][nm]['QLIKE'])}$" for w in windows)
        dm = " & ".join("--" if nm == "GARCH-G" else f"${res[w][nm]['DM']:+.2f}$" for w in windows)
        lines.append(f"{TEX[nm]} & {q} & {dm} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (Path(out) / "tab_robust_window.tex").write_text("\n".join(lines))
    print("  wrote tab_robust_window.tex")
    return res


# ---------------- Check 2: alternative regime classifiers ----------------
def nber_turbulent(dates_oos):
    d = pd.to_datetime(dates_oos); flag = np.zeros(len(d), bool)
    for s, e in NBER:
        flag |= (d >= pd.Timestamp(s)) & (d <= pd.Timestamp(e))
    return flag


def vix_turbulent(dates_all, n_in, dates_oos, vix_csv=None):
    if vix_csv:
        v = pd.read_csv(vix_csv)
        v.columns = [c.lower() for c in v.columns]
        v = v.rename(columns={"date": "Date", "close": "Close", "adj close": "Close"})
    else:
        import yfinance as yf
        vv = yf.download("^VIX", start="2000-01-01", end="2025-01-01", auto_adjust=False, progress=False)
        if isinstance(vv.columns, pd.MultiIndex):
            vv.columns = [c[0] for c in vv.columns]
        v = pd.DataFrame({"Date": vv.index, "Close": vv["Close"].to_numpy()})
    vs = pd.Series(pd.to_numeric(v["Close"], errors="coerce").to_numpy(),
                   index=pd.to_datetime(v["Date"]))
    in_vix = vs.reindex(pd.to_datetime(dates_all)).to_numpy()[:n_in]
    thr = np.nanpercentile(in_vix[~np.isnan(in_vix)], 80)           # top quintile in-sample
    oos_vix = vs.reindex(pd.to_datetime(dates_oos)).to_numpy()
    return oos_vix > thr, float(thr)


def check_regime(r, dates, window, out, vix_csv=None):
    print("  estimating 1,500-day forecasts ...", flush=True)
    lo = qlike_losses(forecasts(r, window), r, window)
    dates_oos = dates[window:]
    classifiers = [("Rolling-SD ($70^{\\text{th}}$ pct, main)", classify(r, window)[0][window:]),
                   ("NBER recession", nber_turbulent(dates_oos))]
    try:
        vt, thr = vix_turbulent(dates, window, dates_oos, vix_csv)
        classifiers.append(("VIX top quintile", vt))
        print(f"  VIX in-sample 80th pct = {thr:.2f}")
    except Exception as e:
        print(f"  VIX classifier skipped ({e}); pass --vix vix.csv to include it")
    lines = [r"\begin{tabular}{lccc}", r"\hline",
             r"Classifier & Turbulent days & $\Delta$ QLIKE (Beta-$t$-EGARCH $-$ GARCH-G) & DM \\", r"\hline"]
    for label, turb in classifiers:
        g = lo["GARCH-G"][turb]; b = lo["Beta-t-EGARCH"][turb]
        dq = float(b.mean() - g.mean())                 # negative -> Beta beats GARCH-G on turbulent days
        dm = diebold_mariano(g, b)["DM"]                # signed vs GARCH-G, like the main table
        lines.append(f"{label} & ${int(turb.sum())}$ & ${dq:+.4f}$ & ${dm:+.2f}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (Path(out) / "tab_robust_regime.tex").write_text("\n".join(lines))
    print("  wrote tab_robust_regime.tex")


# ---------------- Check 3: weekly vs daily refit ----------------
def check_refit(r, window, out):
    print("  weekly-refit forecasts ...", flush=True)
    weekly = forecasts(r, window)
    daily = dict(weekly)                                # GARCH refit cadence is unchanged
    for nm in ("t-GAS", "Beta-t-EGARCH"):
        print(f"  daily-refit {nm} ...", flush=True)
        cls = next(c for c in MODELS if c.name == nm)
        daily[nm] = rolling(r, cls, window, 1, n_restarts=3)
    lw = qlike_losses(weekly, r, window); ld = qlike_losses(daily, r, window)
    bw, bd = lw["GARCH-G"], ld["GARCH-G"]
    lines = [r"\begin{tabular}{lcccc}", r"\hline",
             r" & \multicolumn{2}{c}{Weekly refit (main)} & \multicolumn{2}{c}{Daily refit} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & QLIKE & DM vs.\ GARCH-G & QLIKE & DM vs.\ GARCH-G \\", r"\hline"]
    for nm in NAMES:
        dmw = "--" if nm == "GARCH-G" else f"${diebold_mariano(bw, lw[nm])['DM']:+.2f}$"
        dmd = "--" if nm == "GARCH-G" else f"${diebold_mariano(bd, ld[nm])['DM']:+.2f}$"
        lines.append(f"{TEX[nm]} & ${_f(lw[nm].mean())}$ & {dmw} & ${_f(ld[nm].mean())}$ & {dmd} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (Path(out) / "tab_robust_refit.tex").write_text("\n".join(lines))
    print("  wrote tab_robust_refit.tex")


# ---------------- Verify the three results-paragraph conclusions ----------------
def verify_claims(r, window):
    """Check the conclusions stated in the results-section robustness paragraph."""
    fc = forecasts(r, window)
    lo = qlike_losses(fc, r, window)
    r_oos = r[window:]
    turb = classify(r, window)[0][window:]
    print("\nResults-paragraph claim checks (main 1,500-day run):")
    for label, idx in [("calm", ~turb), ("turbulent", turb)]:
        L = np.column_stack([lo[nm][idx] for nm in NAMES])
        p = mcs(L, NAMES)["pvalues"]
        keep = [nm for nm in NAMES if p[nm] >= 0.10]
        flag = "retained" if p["GARCH-G"] >= 0.10 else "EXCLUDED"
        print(f"  (1) MCS {label:9s}: GARCH-Gauss p={p['GARCH-G']:.2f} ({flag}); set = {keep}")
    for label, idx in [("calm", ~turb), ("turbulent", turb)]:
        dm = diebold_mariano(lo["GARCH-G"][idx], lo["GARCH-t"][idx])["DM"]
        print(f"  (2) DM GARCH-t vs GARCH-G ({label}) = {dm:+.2f}  (expect - in calm, + in turbulent)")
    for nm in ("t-GAS", "Beta-t-EGARCH"):
        rate = float((r_oos < -fc[nm]["var01"]).mean())
        print(f"  (3) 1% VaR rate {nm:14s} = {100*rate:.2f}%  (target 1.00%; expect over-violation)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--window", type=int, default=1500)
    ap.add_argument("--vix", default=None, help="CSV with Date,Close for ^VIX (optional)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dates, r = load_returns(a.csv)
    print("Check 1: estimation-window length"); check_window(r, a.out)
    print("Check 2: alternative regime classifiers"); check_regime(r, dates, a.window, a.out, a.vix)
    print("Check 3: weekly vs daily refit"); check_refit(r, a.window, a.out)
    verify_claims(r, a.window)
    print(f"\nDone. tab_robust_window.tex, tab_robust_regime.tex, tab_robust_refit.tex in {a.out}/")


if __name__ == "__main__":
    main()
