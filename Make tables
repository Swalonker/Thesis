"""
Turn output/insample.json and output/results.json (written by run_all.py) into
the LaTeX tables used in the thesis. Run after run_all.py:

    python make_tables.py --out output

Writes output/tab_*.tex which you can \\input directly.
"""
import argparse, json
from pathlib import Path

NAMES = ["GARCH-G", "GARCH-t", "t-GAS", "Beta-t-EGARCH"]
TEX = {"GARCH-G": "GARCH-Gauss", "GARCH-t": "GARCH-$t$", "t-GAS": "t-GAS", "Beta-t-EGARCH": "Beta-$t$-EGARCH"}


def f(x, d=4):
    try:
        return f"{x:.{d}f}"
    except (TypeError, ValueError):
        return "--"


def tab_estimates(ins, out):
    rows = []
    for nm in NAMES:
        p = ins[nm]["params"]; se = ins[nm]["se"]
        rows.append((nm, p, se))
    lines = [r"\begin{tabular}{lcccc}", r"\hline",
             " & " + " & ".join(TEX[n] for n in NAMES) + r" \\", r"\hline"]
    for key in ("omega", "alpha", "A", "A_minus", "beta", "B", "nu"):
        if not any(key in ins[n]["params"] for n in NAMES):
            continue
        cells = []
        for n in NAMES:
            p = ins[n]["params"]
            cells.append(f"${f(p[key])}$" if key in p else "--")
        lines.append(f"{key} & " + " & ".join(cells) + r" \\")
        secells = []
        for n in NAMES:
            se = ins[n]["se"]
            secells.append(f"$({f(se[key],3)})$" if key in se else "")
        lines.append(" & " + " & ".join(secells) + r" \\")
    lines.append(r"\hline")
    for stat in ("loglik", "AIC", "BIC"):
        lines.append(f"{stat} & " + " & ".join(f"${f(ins[n][stat],1)}$" for n in NAMES) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (Path(out) / "tab_estimates.tex").write_text("\n".join(lines))


def tab_full_qlike(res, out):
    full = res["full"]
    lines = [r"\begin{tabular}{lcccc}", r"\hline",
             r" & QLIKE & DM vs.\ GARCH-G & DM $p$ & MCS$_{0.10}$ \\", r"\hline"]
    for n in NAMES:
        d = full[n]
        dm = "--" if n == "GARCH-G" else f"${d['DM']:+.2f}$"
        dmp = "--" if n == "GARCH-G" else f"${f(d['DM_p'],2)}$"
        lines.append(f"{TEX[n]} & ${f(d['QLIKE'])}$ & {dm} & {dmp} & ${f(full['_mcs'][n],2)}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (Path(out) / "tab_full_qlike.tex").write_text("\n".join(lines))


def tab_regime_qlike(res, out):
    calm, tb = res["calm"], res["turbulent"]
    lines = [r"\begin{tabular}{lcccccc}", r"\hline",
             r" & \multicolumn{3}{c}{Calm} & \multicolumn{3}{c}{Turbulent} \\",
             r" & QLIKE & DM & $p$ & QLIKE & DM & $p$ \\", r"\hline"]
    for n in NAMES:
        c, t = calm[n], tb[n]
        cdm = "--" if n == "GARCH-G" else f"${c['DM']:+.2f}$"
        cp = "--" if n == "GARCH-G" else f"${f(c['DM_p'],3)}$"
        tdm = "--" if n == "GARCH-G" else f"${t['DM']:+.2f}$"
        tp = "--" if n == "GARCH-G" else f"${f(t['DM_p'],3)}$"
        lines.append(f"{TEX[n]} & ${f(c['QLIKE'])}$ & {cdm} & {cp} & ${f(t['QLIKE'])}$ & {tdm} & {tp} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (Path(out) / "tab_regime_qlike.tex").write_text("\n".join(lines))


def tab_mcs(res, out):
    lines = [r"\begin{tabular}{lccc}", r"\hline",
             r" & Full sample & Calm & Turbulent \\", r"\hline"]
    for n in NAMES:
        lines.append(f"{TEX[n]} & ${f(res['full']['_mcs'][n],2)}$ & "
                     f"${f(res['calm']['_mcs'][n],2)}$ & ${f(res['turbulent']['_mcs'][n],2)}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (Path(out) / "tab_mcs.tex").write_text("\n".join(lines))


def tab_var(res, out):
    lines = [r"\begin{tabular}{lcccccc}", r"\hline",
             r" & \multicolumn{2}{c}{$p=0.01$} & \multicolumn{2}{c}{$p=0.05$} & rate01 & rate05 \\",
             r" & UC & CC & UC & CC & & \\", r"\hline"]
    for n in NAMES:
        d = res["full"][n]
        lines.append(f"{TEX[n]} & ${f(d['UC01'],2)}$ & ${f(d['CC01'],2)}$ & "
                     f"${f(d['UC05'],2)}$ & ${f(d['CC05'],2)}$ & "
                     f"${100*d['rate01']:.2f}\\%$ & ${100*d['rate05']:.2f}\\%$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (Path(out) / "tab_var_full.tex").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="output")
    out = ap.parse_args().out
    ins = json.load(open(Path(out) / "insample.json"))
    res = json.load(open(Path(out) / "results.json"))
    tab_estimates(ins, out); tab_full_qlike(res, out); tab_regime_qlike(res, out)
    tab_mcs(res, out); tab_var(res, out)
    print(f"Wrote tab_*.tex to {out}/")


if __name__ == "__main__":
    main()
