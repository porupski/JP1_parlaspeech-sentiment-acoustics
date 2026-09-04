# ============================================================
# Script:  latex_utils.py
# Release: 1.0
# Version: v1.00
# Purpose: Format results as LaTeX table strings; emit numbers.json keys
# ============================================================

import numpy as np
import json
from pathlib import Path
from utils.stats import format_p


LANG_ORDER = ["HR", "CZ", "PL", "RS", "SI"]
FEATURE_DISPLAY = {
    "f0_raw": "F0 (pitch)",
    "f0_norm": "F0 (norm.)",
    "intensity_raw": "Intensity",
    "intensity_norm": "Intensity (norm.)",
    "speechrate_wps": "Speech rate",
    "speechrate_sps": "Speech rate (syl)",
}


def _fmt(val, decimals: int = 3, na: str = "—") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return f"{val:.{decimals}f}"


def _fmt_p(p: float, corrected_p: float = None) -> str:
    """Format p-value cell. If corrected_p provided, show both."""
    raw = format_p(p) if not np.isnan(p) else "—"
    if corrected_p is not None and not np.isnan(corrected_p):
        corr = format_p(corrected_p)
        return f"$p {raw}$ / $p_{{BH}} {corr}$"
    return f"$p {raw}$"


def format_table_h1(results: dict, features: list[str],
                    languages: list[str] = None) -> str:
    """
    Table 2: H1 Wilcoxon results.
    results keyed by (lang, feat) → {'speaker_avg': {p, rbc, concordance}, 'utterance_level': ...}
    """
    if languages is None:
        languages = LANG_ORDER
    lines = [
        r"\begin{table}[h!]",
        r"\centering\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l|ccc|ccc}",
        r"\hline",
        r"Feature & \multicolumn{3}{c|}{Speaker Average} & \multicolumn{3}{c}{Utterance Level} \\",
        r" & $p$ & RBC & P(N>P) & $p$ & RBC & P(N>P) \\",
        r"\hline",
    ]
    for lang in languages:
        lines.append(r"\hline")
        lines.append(r"\multicolumn{7}{l}{\normalsize\textbf{" + lang + r"}} \\")
        lines.append(r"\hline")
        for feat in features:
            key = (lang, feat)
            r = results.get(key, {})
            sa = r.get("speaker_avg", {})
            ul = r.get("utterance_level", {})
            row = (
                FEATURE_DISPLAY.get(feat, feat)
                + f" & {_fmt_p(sa.get('p', np.nan))}"
                + f" & {_fmt(sa.get('rbc'), 3)}"
                + f" & {_fmt(sa.get('concordance'), 3)}"
                + f" & {_fmt_p(ul.get('p', np.nan))}"
                + f" & {_fmt(ul.get('rbc'), 3)}"
                + f" & {_fmt(ul.get('concordance'), 3)}"
                + r" \\"
            )
            lines.append(row)
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{H1: Wilcoxon signed-rank comparing negative vs positive sentiment.}",
              r"\label{tab:h1_wilcoxon}", r"\end{table}"]
    return "\n".join(lines)


def format_table_h2(results: dict, features: list[str],
                    languages: list[str] = None) -> str:
    """
    Table 3: H2 Kendall tau results.
    results keyed by (lang, feat) → {p, mean_tau, ci_lo, ci_hi, n_speakers, n_sig_speakers}
    """
    if languages is None:
        languages = LANG_ORDER
    lines = [
        r"\begin{table}[h!]",
        r"\centering\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l|cccc}",
        r"\hline",
        r"Feature & $p$ & $\bar{\tau}$ [95\% CI] & Sig.\ speakers \\",
        r"\hline",
    ]
    for lang in languages:
        lines.append(r"\hline")
        lines.append(r"\multicolumn{4}{l}{\normalsize\textbf{" + lang + r"}} \\")
        lines.append(r"\hline")
        for feat in features:
            key = (lang, feat)
            r = results.get(key, {})
            p = r.get("p", np.nan)
            tau = r.get("mean_tau", np.nan)
            ci_lo = r.get("ci_lo", np.nan)
            ci_hi = r.get("ci_hi", np.nan)
            n = r.get("n_speakers", "?")
            n_sig = r.get("n_sig_speakers", "?")
            tau_str = f"{_fmt(tau, 3)} [{_fmt(ci_lo, 3)}, {_fmt(ci_hi, 3)}]"
            sig_str = f"{n_sig}/{n}"
            row = (
                FEATURE_DISPLAY.get(feat, feat)
                + f" & {_fmt_p(p)}"
                + f" & {tau_str}"
                + f" & {sig_str}"
                + r" \\"
            )
            lines.append(row)
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{H2: Kendall's $\tau$ across the full sentiment continuum (per-speaker, then one-sample $t$-test). CI = bootstrap 95\% CI on mean $\tau$.}",
              r"\label{tab:h2_kendall}", r"\end{table}"]
    return "\n".join(lines)


def format_table_h3(results: dict, features: list[str],
                    languages: list[str] = None) -> str:
    """Table 4: H3 split analysis results."""
    if languages is None:
        languages = LANG_ORDER
    lines = [
        r"\begin{table}[h!]",
        r"\centering\small\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{l|cccc|cccc|c}",
        r"\hline",
        r"Feature & \multicolumn{4}{c|}{Negative side} & \multicolumn{4}{c|}{Positive side} & Check \\",
        r" & $p_K$ & $\tau$ & $p_L$ & Slope & $p_K$ & $\tau$ & $p_L$ & Slope & \\",
        r"\hline",
    ]
    SYM = {"strong": r"$\checkmark$", "partial": r"$+$", "none": r"$\times$"}
    for lang in languages:
        lines.append(r"\hline")
        lines.append(r"\multicolumn{10}{l}{\normalsize\textbf{" + lang + r"}} \\")
        lines.append(r"\hline")
        for feat in features:
            key = (lang, feat)
            r = results.get(key, {})
            neg = r.get("neg", {})
            pos = r.get("pos", {})
            chk = r.get("check", "none")
            row = (
                FEATURE_DISPLAY.get(feat, feat)
                + f" & {_fmt_p(neg.get('kendall_p', np.nan))}"
                + f" & {_fmt(neg.get('kendall_tau'), 3)}"
                + f" & {_fmt_p(neg.get('linear_p', np.nan))}"
                + f" & {_fmt(neg.get('linear_slope'), 3)}"
                + f" & {_fmt_p(pos.get('kendall_p', np.nan))}"
                + f" & {_fmt(pos.get('kendall_tau'), 3)}"
                + f" & {_fmt_p(pos.get('linear_p', np.nan))}"
                + f" & {_fmt(pos.get('linear_slope'), 3)}"
                + f" & {SYM.get(chk, '?')}"
                + r" \\"
            )
            lines.append(row)
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{H3: Split-point analysis. $p_K$ = Kendall $p$-value; $p_L$ = linear regression $p$-value. Check: $\checkmark$ strong arousal pattern, $+$ partial, $\times$ none.}",
              r"\label{tab:h3_split}", r"\end{table}"]
    return "\n".join(lines)


def write_table(tex: str, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(tex)


def build_numbers_json(h1: dict, h2: dict, h3: dict,
                        features: list[str], languages: list[str]) -> dict:
    """
    Build flat numbers.json dict. Every key is an in-text value.
    Format: {metric}_{feature}_{lang} or aggregate counts.
    """
    numbers = {}

    # H1 counts
    h1_sig_sa = sum(
        1 for lang in languages for feat in features
        if h1.get((lang, feat), {}).get("speaker_avg", {}).get("p", 1.0) < 0.05
    )
    numbers["h1_sig_speaker_avg"] = h1_sig_sa
    numbers["h1_total"] = len(features) * len(languages)

    # H2 counts
    h2_sig = sum(
        1 for lang in languages for feat in features
        if h2.get((lang, feat), {}).get("p", 1.0) < 0.05
    )
    numbers["h2_sig"] = h2_sig
    numbers["h2_total"] = len(features) * len(languages)

    # H3 counts
    h3_strong = sum(
        1 for lang in languages for feat in features
        if h3.get((lang, feat), {}).get("check") == "strong"
    )
    h3_partial = sum(
        1 for lang in languages for feat in features
        if h3.get((lang, feat), {}).get("check") == "partial"
    )
    numbers["h3_strong"] = h3_strong
    numbers["h3_partial"] = h3_partial
    numbers["h3_supported"] = h3_strong + h3_partial
    numbers["h3_total"] = len(features) * len(languages) * 2  # 2 per combo (K + L)

    # per-feature/language tau values
    for lang in languages:
        for feat in features:
            r = h2.get((lang, feat), {})
            key = f"tau_{feat}_{lang}"
            numbers[key] = r.get("mean_tau")
            numbers[f"tau_ci_lo_{feat}_{lang}"] = r.get("ci_lo")
            numbers[f"tau_ci_hi_{feat}_{lang}"] = r.get("ci_hi")

    return numbers


def write_numbers_json(numbers: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(numbers, f, indent=2, default=lambda x: None if x is None else float(x))
