"""Validazione dei dati sintetici: qualita' statistica e privacy.

- Correlazione: differenza tra le matrici di correlazione reale/sintetica.
- Distribuzione: test di Kolmogorov-Smirnov per le numeriche, Total
  Variation Distance per le categoriche, piu' un confronto delle statistiche.
- Privacy: ricerca di record sintetici identici a quelli reali e calcolo
  della Distance to Closest Record (DCR), per individuare overfitting.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from .data_engine import DataPreprocessor

# ------------------------------------------------------------- correlazione


def correlation_matrices(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = real.select_dtypes("number").columns.intersection(synth.columns)
    return real[cols].corr(), synth[cols].corr()


def correlation_report(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    corr_r, corr_s = correlation_matrices(real, synth)
    diff = (corr_r - corr_s).abs()
    mask = ~np.eye(len(diff), dtype=bool)
    off_diag = diff.where(mask).stack()
    return {
        "mean_abs_diff": float(off_diag.mean()) if len(off_diag) else 0.0,
        "max_abs_diff": float(off_diag.max()) if len(off_diag) else 0.0,
        "real": corr_r.round(3).to_dict(),
        "synthetic": corr_s.round(3).to_dict(),
    }


def plot_correlation_heatmaps(real: pd.DataFrame, synth: pd.DataFrame):
    corr_r, corr_s = correlation_matrices(real, synth)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    kw = dict(vmin=-1, vmax=1, cmap="RdBu_r", annot=True, fmt=".2f", annot_kws={"size": 7})
    sns.heatmap(corr_r, ax=axes[0], **kw)
    axes[0].set_title("Dati reali")
    sns.heatmap(corr_s, ax=axes[1], **kw)
    axes[1].set_title("Dati sintetici")
    sns.heatmap((corr_r - corr_s).abs(), ax=axes[2], vmin=0, vmax=1, cmap="Reds",
                annot=True, fmt=".2f", annot_kws={"size": 7})
    axes[2].set_title("|Differenza|")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- distribuzione


def distribution_report(real: pd.DataFrame, synth: pd.DataFrame) -> list[dict]:
    rows = []
    for col in real.columns:
        r, s = real[col].dropna(), synth[col].dropna()
        if pd.api.types.is_numeric_dtype(r) and r.nunique() > 2:
            ks = ks_2samp(r, s)
            rows.append({
                "column": col,
                "type": "numeric",
                "real_mean": float(r.mean()),
                "synth_mean": float(s.mean()),
                "real_std": float(r.std()),
                "synth_std": float(s.std()),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                # 1 - KS: 1 = distribuzioni identiche
                "similarity": float(1 - ks.statistic),
            })
        else:
            pr = r.astype(str).value_counts(normalize=True)
            ps = s.astype(str).value_counts(normalize=True)
            idx = pr.index.union(ps.index)
            tvd = 0.5 * (pr.reindex(idx, fill_value=0) - ps.reindex(idx, fill_value=0)).abs().sum()
            rows.append({
                "column": col,
                "type": "categorical",
                "real_freq": pr.round(3).to_dict(),
                "synth_freq": ps.round(3).to_dict(),
                "tvd": float(tvd),
                "similarity": float(1 - tvd),
            })
    return rows


def plot_distributions(real: pd.DataFrame, synth: pd.DataFrame, columns: list[str] | None = None):
    columns = columns or list(real.columns)
    n = len(columns)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, columns):
        r, s = real[col].dropna(), synth[col].dropna()
        if pd.api.types.is_numeric_dtype(r) and r.nunique() > 2:
            bins = np.histogram_bin_edges(pd.concat([r, s]), bins=30)
            ax.hist(r, bins=bins, alpha=0.55, density=True, label="Reale")
            ax.hist(s, bins=bins, alpha=0.55, density=True, label="Sintetico")
        else:
            freq = pd.DataFrame({
                "Reale": r.astype(str).value_counts(normalize=True),
                "Sintetico": s.astype(str).value_counts(normalize=True),
            }).fillna(0)
            freq.plot.bar(ax=ax, rot=0, alpha=0.8)
        ax.set_title(col)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ privacy


def privacy_report(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    preprocessor: DataPreprocessor | None = None,
) -> dict:
    """Controlla che il modello non abbia "memorizzato" i dati reali.

    - exact_matches: righe sintetiche identiche a una riga reale;
    - DCR (Distance to Closest Record): distanza di ogni record sintetico
      dal record reale piu' vicino, nello spazio normalizzato;
    - come riferimento calcolo la stessa distanza tra i record reali stessi
      (escludendo se stessi): se i sintetici sono molto piu' vicini ai reali
      di quanto i reali lo siano tra loro, c'e' rischio di copia.
    """
    cols = list(real.columns)
    real_keys = set(map(tuple, real[cols].astype(str).to_numpy()))
    synth_keys = synth[cols].astype(str).to_numpy()
    exact = int(sum(tuple(row) in real_keys for row in synth_keys))

    if preprocessor is not None:
        xr = preprocessor.transform(real)
        xs = preprocessor.transform(synth)
    else:
        num = real.select_dtypes("number").columns
        scaler = MinMaxScaler().fit(real[num].fillna(real[num].median()))
        xr = scaler.transform(real[num].fillna(real[num].median()))
        xs = scaler.transform(synth[num].fillna(real[num].median()))

    nn_real = NearestNeighbors(n_neighbors=2).fit(xr)
    dcr_synth = nn_real.kneighbors(xs, n_neighbors=1)[0][:, 0]
    dcr_real = nn_real.kneighbors(xr, n_neighbors=2)[0][:, 1]  # salto se stesso

    threshold = float(np.percentile(dcr_real, 5))
    too_close = float(np.mean(dcr_synth < threshold))

    return {
        "n_real": len(real),
        "n_synthetic": len(synth),
        "exact_matches": exact,
        "exact_match_rate": exact / max(1, len(synth)),
        "dcr_synth_mean": float(dcr_synth.mean()),
        "dcr_synth_min": float(dcr_synth.min()),
        "dcr_real_mean": float(dcr_real.mean()),
        "dcr_real_p5": threshold,
        "share_closer_than_real_p5": too_close,
        # Valutazione semplice: nessuna copia esatta e meno del 10% dei
        # record sintetici "sospettosamente" vicino a un reale.
        "passed": bool(exact == 0 and too_close < 0.10),
    }


# ---------------------------------------------------------------- riepilogo


def full_report(real: pd.DataFrame, synth: pd.DataFrame, preprocessor: DataPreprocessor | None = None) -> dict:
    dist = distribution_report(real, synth)
    corr = correlation_report(real, synth)
    priv = privacy_report(real, synth, preprocessor)
    fidelity = float(np.mean([d["similarity"] for d in dist])) if dist else 0.0
    return {
        "fidelity_score": fidelity,
        "correlation": corr,
        "distribution": dist,
        "privacy": priv,
    }


def comparison_table(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """Tabella comparativa reali vs sintetici (statistiche descrittive)."""
    num = real.select_dtypes("number").columns
    r = real[num].describe().T[["mean", "std", "min", "50%", "max"]]
    s = synth[num].describe().T[["mean", "std", "min", "50%", "max"]]
    table = pd.concat({"Reale": r, "Sintetico": s}, axis=1).swaplevel(axis=1)
    return table.sort_index(axis=1, level=0, sort_remaining=False).round(3)
