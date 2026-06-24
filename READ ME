# Robust Score-Driven Time Series Models — replication code

Reproduces every empirical table in the thesis (in-sample estimates,
full-sample QLIKE / Diebold–Mariano / Model Confidence Set, the VaR backtests,
and the regime-conditional results) from one command.

## Models
`GARCH-G`, `GARCH-t` (Bollerslev 1987), `t-GAS` (Creal–Koopman–Lucas 2013, symmetric
log-variance), `Beta-t-EGARCH` (Harvey 2013, asymmetric). All on daily S&P 500
log-returns in percentage points, zero conditional mean.

## How to run
```bash
pip install -r requirements.txt
python fetch_data.py                       # -> sp500_returns.csv  (Date, AdjClose, r)
python run_all.py --csv sp500_returns.csv --out output --window 1500
python make_tables.py --out output         # -> output/tab_*.tex
```
`run_all.py` writes `output/insample.json` and `output/results.json`; `make_tables.py`
turns those into `tab_estimates.tex`, `tab_full_qlike.tex`, `tab_regime_qlike.tex`,
`tab_mcs.tex`, `tab_var_full.tex`, which you can `\input` into the thesis.

## Files
| file | purpose |
|------|---------|
| `fetch_data.py` | download ^GSPC, write the single canonical returns CSV |
| `models.py` | the four volatility models (uniform `fit`/`filter`/`forecast_next` API) |
| `evaluation.py` | QLIKE, Diebold–Mariano, MCS, Kupiec/Christoffersen/DQ |
| `run_all.py` | rolling 1,500-day forecast + full-sample and regime evaluation |
| `make_tables.py` | JSON results -> LaTeX tables |

## Design notes (what this fixes vs. the earlier scripts)
- **QLIKE on the conditional variance.** For the score-driven models the filtered
  `sigma2` is the Student-*t* *scale*; the variance is `scale * nu/(nu-2)`. QLIKE and
  the loss differentials are computed on the variance for all four models.
- **VaR quantiles.** Gaussian uses `Phi^{-1}`; GARCH-*t* uses `sqrt((nu-2)/nu)*t^{-1}`
  (unit-variance standardised *t*); t-GAS / Beta-*t*-EGARCH use `t^{-1}` on the scale.
- **MCS p-value alignment.** `arch.bootstrap.MCS` returns p-values ordered by
  elimination, not by model. `evaluation.mcs` realigns each p-value to its own model
  via the result index. (Zipping positionally inverts the table — the bug that put
  Beta-*t*-EGARCH at p = 1.00 and rejected the GARCH benchmarks.)
- **Numerical stability.** The score-driven log-variance recursion is clipped to
  `[-20, 20]` to prevent `exp()` overflow.
- **NumPy 2.x.** Uses `scipy.special.gammaln`; the old `np.math.gamma` (removed in
  NumPy 2.0) is gone.

## Reproducibility caveat
The script that originally produced the saved `Data/forecasts/*.csv` could not be
located, and the earlier `run_experiment.py` did **not** match the thesis methodology
(it evaluated QLIKE on the scale, used uncorrected GARCH-*t* VaR, and never computed the
regime tables). This pipeline is rebuilt to the methodology and is the authoritative
source going forward: re-run it and update any thesis number that has shifted so the
repository and the thesis agree.

The appendix robustness checks (1,000/2,000-day windows; NBER and VIX regime
classifiers; daily refit) are not yet automated here — the window-length and refit
variants are one-line changes to `run_all.py`; the NBER/VIX classifiers need their own
external series.
