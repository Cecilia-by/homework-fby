"""Recalculate published summaries and tests from the authors' CSV outputs."""

from pathlib import Path

import matplotlib
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "published"
OUT = ROOT / "results" / "published"
TIME_ATTACKS = ("correlated_signal", "max_engine_coolant", "max_speedometer",
                "reverse_light_off", "reverse_light_on")


matplotlib.use("Agg")
from matplotlib import pyplot as plt


def heatmap(rf_data):

    labels = ("Correlated signal", "Engine coolant", "Speedometer", "Reverse light off",
              "Reverse light on")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
    fig, axs = plt.subplots(2, 5, figsize=(12, 4.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.89, bottom=.14, top=.85, wspace=.13, hspace=.14)
    for i, (mode, title) in enumerate((("embeddings_only", "Graph only"),
                                        ("normalized", "Graph + signals"))):
        for j, (ax, data) in enumerate(zip(axs[i], rf_data)):
            grid = data[data.version == mode].pivot(index="window_size", columns="offset",
                                                     values="roc_auc")
            grid = grid.reindex(index=range(2, 16), columns=range(1, 16))
            assert grid.count().sum() == 119 and grid.stack().between(.8, 1).all()
            img = ax.imshow(grid, origin="lower", extent=(.5, 15.5, 1.5, 15.5),
                            interpolation="none", cmap="viridis", vmin=.8, vmax=1)
            ax.set_facecolor("#efefef")
            ax.set_xticks([1, 5, 10, 15])
            ax.set_yticks([2, 5, 10, 15])
            if i == 0:
                ax.set_title(labels[j])
            if j == 0:
                ax.set_ylabel(title, fontweight="bold")
    fig.suptitle("Published ROAD results: random forest", fontsize=15, fontweight="bold", y=.97)
    fig.supxlabel("Window step (s)", y=.03)
    fig.supylabel("Window size (s)", x=.012)
    cax = fig.add_axes([.91, .19, .016, .62])
    fig.colorbar(img, cax=cax, label="ROC-AUC", ticks=[.8, .85, .9, .95, 1])
    fig.savefig(OUT / "rf-time-heatmap.png", dpi=190)
    plt.close(fig)


def summarize(data, setting, scenario, model, baseline, combined):
    baseline_auc = data.loc[data.version == baseline, "roc_auc"]
    combined_auc = data.loc[data.version == combined, "roc_auc"]
    if len(baseline_auc) != len(combined_auc) or not len(baseline_auc):
        raise ValueError(f"Missing paired configuration counts for {scenario}")
    u = mannwhitneyu(combined_auc, baseline_auc, alternative="greater")
    ks = ks_2samp(combined_auc, baseline_auc)
    notebook_p = ks.pvalue / 2 if combined_auc.mean() > baseline_auc.mean() else 1 - ks.pvalue / 2
    directional = ks_2samp(combined_auc, baseline_auc, alternative="less")
    return dict(setting=setting, scenario=scenario, model=model,
                configurations=len(baseline_auc), graph_mean=baseline_auc.mean(),
                graph_sd=baseline_auc.std(ddof=0), graph_median=baseline_auc.median(),
                graph_min=baseline_auc.min(), graph_max=baseline_auc.max(),
                hybrid_mean=combined_auc.mean(), hybrid_sd=combined_auc.std(ddof=0),
                hybrid_median=combined_auc.median(), hybrid_min=combined_auc.min(),
                hybrid_max=combined_auc.max(), mann_whitney_u=u.statistic,
                mann_whitney_p=u.pvalue, ks_stat=ks.statistic, ks_notebook_p=notebook_p,
                ks_directional_stat=directional.statistic,
                ks_directional_p=directional.pvalue)


def main():
    rows, rf_data = [], []
    for kind in TIME_ATTACKS:
        for model in ("rf", "xgb"):
            stem = kind
            if model == "xgb":
                stem += "_temp" if kind == "max_engine_coolant" else ""
                stem += "_xgboost"
            data = pd.read_csv(DATA / f"performance_metrics_{stem}.csv")
            rows.append(summarize(data, "seconds", kind, model, "embeddings_only", "normalized"))
            if model == "rf":
                rf_data.append(data)
    sample = pd.read_csv(DATA / "final_consolidated_performance_metrics.csv")
    for kind, g in sample.groupby("attack_type"):
        rows.append(summarize(g, "frames", kind.removesuffix("_attack"), "published",
                              "embeddings_only_normalized", "all_normalized"))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "roc-summary-and-tests.csv", index=False, float_format="%.12g")
    heatmap(rf_data)
    print(out[["setting", "scenario", "model", "configurations", "graph_mean",
               "hybrid_mean", "mann_whitney_u", "mann_whitney_p", "ks_stat",
               "ks_notebook_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
