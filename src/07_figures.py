#!/usr/bin/env python3
"""Generate all figures for the GeoRAG Member 2 model-upgrade work.

Inputs (read-only): experiment JSONs, results/model_comparison.csv,
prediction rasters, change-detection rasters / analysis json.
Output: outputs/figures/*.png
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FIG_DIR = os.path.join(C.ROOT, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_LABELS = {"random_forest": "RF (baseline)", "xgboost": "XGB (baseline)",
                "dla": "dla-lite (new)", "mts": "mts-lite (new)"}
MODEL_FILES = {"random_forest": "random_forest", "xgboost": "xgboost",
               "dla": "dla-lite-adapted", "mts": "mts-lite-adapted"}
COLORS = {"random_forest": "#4C72B0", "xgboost": "#55A868",
          "dla": "#C44E52", "mts": "#8172B2"}
EXPERIMENTS = ["E0_2015", "E0_2020", "E0_2025", "E1_2015_to_2020",
               "E2_2015_to_2025", "E3_2015_2020_to_2025"]
TEMPORAL = EXPERIMENTS[3:]
# native label order used by evaluation (native ids 1..4)
NATIVE_AXIS = ["1 Water\n(out 3)", "2 Urban\n(out 1 Built-up)",
               "3 Soil\n(out 4 Open Land)", "4 Vegetation\n(out 2)"]
CLASS_COLORS = ["#E74C3C", "#27AE60", "#3498DB", "#B7955B"]  # out 1..4


def load_all():
    data = {}
    for p in sorted(glob.glob(os.path.join(C.ROOT, "experiments", "baselines",
                                           "*.json"))):
        d = C.load_json(p)
        for k, m in d["models"].items():
            data[(d["experiment"], k)] = m
    for p in sorted(glob.glob(os.path.join(C.ROOT, "experiments", "transformers",
                                           "*_seed*.json"))):
        d = C.load_json(p)
        data[(d["experiment"], d["model"]["kind"])] = d["results"]
    return data


def fig_macro_f1(data):
    models = ["random_forest", "xgboost", "dla", "mts"]
    present = [m for m in models if any(k == m for (_, k) in data)]
    x = np.arange(len(EXPERIMENTS))
    w = 0.8 / max(len(present), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(present):
        vals = [data.get((e, m), {}).get("f1_macro", np.nan) for e in EXPERIMENTS]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=MODEL_LABELS[m],
               color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", "\n") for e in EXPERIMENTS], fontsize=8)
    ax.set_ylabel("Macro F1 (measured)")
    ax.set_title("Model comparison — macro F1 by experiment (seed 42, "
                 "identical splits/metrics)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_macro_f1_comparison.png"), dpi=140)
    plt.close(fig)


def fig_confusions_temporal(data):
    models = [m for m in ["random_forest", "xgboost", "dla", "mts"]
              if any(k == m for (_, k) in data)]
    fig, axes = plt.subplots(len(models), len(TEMPORAL),
                             figsize=(10.5, 3.1 * len(models)), squeeze=False)
    for r, m in enumerate(models):
        for c, e in enumerate(TEMPORAL):
            ax = axes[r][c]
            res = data.get((e, m))
            if not res:
                ax.axis("off")
                continue
            cm = np.array(res["confusion_matrix"], dtype=float)
            row_n = cm / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(row_n, cmap="Blues", vmin=0, vmax=1)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{cm[i, j]:,.0f}\n{100*row_n[i, j]:.1f}%",
                            ha="center", va="center", fontsize=6.5,
                            color="black" if row_n[i, j] < 0.6 else "white")
            ax.set_xticks(range(4), [n.split("\n")[0] for n in NATIVE_AXIS],
                          fontsize=6.5, rotation=30, ha="right")
            ax.set_yticks(range(4), NATIVE_AXIS, fontsize=6.5)
            if r == 0:
                ax.set_title(e, fontsize=9)
            if c == 0:
                ax.set_ylabel(MODEL_LABELS[m], fontsize=9)
            if r == len(models) - 1:
                ax.set_xlabel("predicted (native ids)", fontsize=7)
    fig.suptitle("Confusion matrices — temporal experiments (rows = true class, "
                 "row-normalized %; counts + %)", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_confusions_temporal.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def fig_per_class_f1(data):
    classes = ["1 Water", "2 Urban", "3 Soil", "4 Vegetation"]
    models = [m for m in ["random_forest", "xgboost", "dla", "mts"]
              if any(k == m for (_, k) in data)]
    x = np.arange(len(classes))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(models):
        vals = []
        for ci, cid in enumerate(["1", "2", "3", "4"]):
            per = [data[(e, m)]["per_class"][cid]["f1"]
                   for e in TEMPORAL if (e, m) in data]
            vals.append(np.mean(per) if per else np.nan)
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=MODEL_LABELS[m],
               color=COLORS[m])
    ax.set_xticks(x, classes)
    ax.set_ylabel("F1 (mean over E1/E2/E3)")
    ax.set_title("Per-class F1 — mean over the three temporal experiments "
                 "(evaluation in Member 1's native class ids)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_per_class_f1_temporal.png"), dpi=140)
    plt.close(fig)


def read_map(path):
    with rasterio.open(path) as src:
        return src.read(1)


def fig_pred_maps():
    cmap = ListedColormap(["#FFFFFF"] + CLASS_COLORS)   # 0=NoData white
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    for key, label in MODEL_FILES.items():
        paths = [os.path.join(C.ROOT, "outputs", "predictions",
                              f"pred_{y}_{label}.tif") for y in (2015, 2020, 2025)]
        if not all(os.path.exists(p) for p in paths):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, p, y in zip(axes, paths, (2015, 2020, 2025)):
            m = read_map(p)
            ax.imshow(m, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_title(f"{y} — {MODEL_LABELS[key]}", fontsize=10)
            ax.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color="#FFFFFF",
                                 edgecolor="gray")] + \
                  [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS]
        fig.legend(handles, ["NoData", "Built-up", "Vegetation", "Water",
                             "Open Land"], loc="lower center", ncol=5, fontsize=9)
        fig.suptitle(f"Prediction maps (output scheme: 1 Built-up, 2 Vegetation, "
                     f"3 Water, 4 Open Land) — {MODEL_LABELS[key]}", fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(os.path.join(FIG_DIR, f"fig_pred_maps_{key}.png"), dpi=140)
        plt.close(fig)


def fig_change_maps():
    # categories: 0 nodata, 1 stable, 2 changed->built-up, 3 other change
    cmap = ListedColormap(["#FFFFFF", "#D5D8DC", "#E74C3C", "#F39C12"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    for key, label in MODEL_FILES.items():
        ad = os.path.join(C.ROOT, "outputs", "change_detection",
                          f"{label}_analysis.json")
        if not os.path.exists(ad):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, (y1, y2) in zip(axes, [(2015, 2020), (2020, 2025)]):
            p = os.path.join(C.ROOT, "outputs", "change_detection", label,
                             f"change_{y1}_{y2}.tif")
            if not os.path.exists(p):
                ax.axis("off")
                continue
            t = read_map(p).astype(int)
            cat = np.zeros_like(t)
            cat[t > 0] = 1
            stable = {1, 6, 11, 16}
            changed = (t > 0) & ~np.isin(t, list(stable))
            cat[changed] = 3
            # id = (from-1)*4 + (to-1) + 1 ; into built-up means to=1 -> ids 1,5,9,13
            # (id 1 = stable built-up, already class 1 above)
            into_bu = np.isin(t, [5, 9, 13])
            cat[into_bu] = 2
            ax.imshow(cat, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_title(f"{y1} → {y2} — {MODEL_LABELS[key]}", fontsize=10)
            ax.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c)
                   for c in ["#D5D8DC", "#E74C3C", "#F39C12", "#FFFFFF"]]
        fig.legend(handles, ["stable cover", "new Built-up (to class 1)",
                             "other land-cover change", "NoData"],
                   loc="lower center", ncol=4, fontsize=9)
        fig.suptitle(f"Change detection 2015→2020 and 2020→2025 — "
                     f"{MODEL_LABELS[key]}", fontsize=11)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        fig.savefig(os.path.join(FIG_DIR, f"fig_change_{key}.png"), dpi=140)
        plt.close(fig)


def fig_transition_heatmaps():
    for key, label in MODEL_FILES.items():
        ad = os.path.join(C.ROOT, "outputs", "change_detection",
                          f"{label}_analysis.json")
        if not os.path.exists(ad):
            continue
        with open(ad) as fh:
            analysis = json.load(fh)
        pairs = analysis["change_pairs"]
        fig, axes = plt.subplots(1, len(pairs), figsize=(10, 4.2),
                                 squeeze=False)
        names = ["Built-up", "Vegetation", "Water", "Open Land"]
        for ax, p in zip(axes[0], pairs):
            mat = np.array(p["transition_area_ha_matrix"])
            im = ax.imshow(mat, cmap="OrRd")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{mat[i, j]:,.0f}", ha="center", va="center",
                            fontsize=7)
            ax.set_xticks(range(4), names, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(4), names, fontsize=8)
            ax.set_title(f"{p['pair']} (ha)", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"Transition areas (geodesic hectares) — "
                     f"{MODEL_LABELS[key]}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(FIG_DIR, f"fig_transition_heatmap_{key}.png"),
                    dpi=140)
        plt.close(fig)


def main():
    data = load_all()
    if data:
        fig_macro_f1(data)
        fig_confusions_temporal(data)
        fig_per_class_f1(data)
    fig_pred_maps()
    fig_change_maps()
    fig_transition_heatmaps()
    files = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(FIG_DIR, "*.png")))
    print("figures:", json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
