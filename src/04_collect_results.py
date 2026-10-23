#!/usr/bin/env python3
"""Collect every experiment JSON into results/model_comparison.csv (+ markdown)
and results/per_class_metrics.csv, plus results/environment.json.

Only actual measured values are used; missing metrics stay empty.
"""
import glob
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

RESULTS = os.path.join(C.ROOT, "results")

BASE_TO_NAME = {"random_forest": "RandomForest (retrained baseline)",
                "xgboost": "XGBoost (retrained baseline)"}
KIND_TO_NAME = {"mts": "mts-lite-adapted (transformer, new)",
                "dla": "dla-lite-adapted (transformer, new)"}


def row_from_baseline(exp_json, model_key):
    d = C.load_json(exp_json)
    m = d["models"][model_key]
    ckpt = m.get("checkpoint")
    if not ckpt and model_key == "xgboost":
        candidate = f"experiments/baselines/xgb_{d['experiment']}.pkl"
        if os.path.exists(os.path.join(C.ROOT, candidate)):
            ckpt = candidate
    if not ckpt:
        ckpt = "(not persisted; see checkpoint_note)"
    return {
        "Model": BASE_TO_NAME[model_key],
        "Model_Key": model_key,
        "Architecture": "gradient-boosted trees" if model_key == "xgboost"
        else "random forest (300 trees)",
        "Training_Years": "+".join(str(y) for y in d["train_years"]),
        "Test_Year": d["test_year"],
        "Experiment": d["experiment"],
        "Accuracy": m["accuracy"],
        "Precision_Macro": m["precision_macro"],
        "Recall_Macro": m["recall_macro"],
        "F1_Macro": m["f1_macro"],
        "Kappa": m["kappa"],
        "Training_Time_s": m["train_time_s"],
        "Inference_Time_s": m["inference_time_s"],
        "Model_Size_Bytes": m["model_size_bytes"],
        "Params": "",
        "Input_Type": d.get("input_type", "tabular_9_features"),
        "Validation_Type": d.get("validation_type", ""),
        "Checkpoint_Path": ckpt,
        "N_Test": m.get("n_samples", d.get("n_test")),
        "Note": m.get("checkpoint_note", ""),
    }, m, d


def row_from_transformer(json_path):
    d = C.load_json(json_path)
    m = d["results"]
    return {
        "Model": KIND_TO_NAME[d["model"]["kind"]],
        "Model_Key": d["model"]["kind"],
        "Architecture": d["model"]["architecture_source"],
        "Training_Years": "+".join(str(y) for y in d["train_years"]),
        "Test_Year": d["test_year"],
        "Experiment": d["experiment"],
        "Accuracy": m["accuracy"],
        "Precision_Macro": m["precision_macro"],
        "Recall_Macro": m["recall_macro"],
        "F1_Macro": m["f1_macro"],
        "Kappa": m["kappa"],
        "Training_Time_s": m["train_time_s"],
        "Inference_Time_s": m["inference_time_s"],
        "Model_Size_Bytes": m["model_size_bytes"],
        "Params": d["model"]["params"],
        "Input_Type": d["input_type"],
        "Validation_Type": d["validation_type"],
        "Checkpoint_Path": m["checkpoint"],
        "N_Test": m.get("n_samples"),
        "Note": f"best_epoch={m['best_epoch']}/{m['epochs_run']}",
    }, m, d


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows, per_class = [], []
    for p in sorted(glob.glob(os.path.join(C.ROOT, "experiments", "baselines", "*.json"))):
        for key in ("random_forest", "xgboost"):
            row, m, d = row_from_baseline(p, key)
            rows.append(row)
            _per_class(per_class, row, m)
    for p in sorted(glob.glob(os.path.join(C.ROOT, "experiments", "transformers",
                                           f"*_seed{C.SEED}.json"))):  # excludes *_spatial.json
        row, m, d = row_from_transformer(p)
        rows.append(row)
        _per_class(per_class, row, m)

    order = {"E0_2015": 0, "E0_2020": 1, "E0_2025": 2, "E1_2015_to_2020": 3,
             "E2_2015_to_2025": 4, "E3_2015_2020_to_2025": 5}
    rows.sort(key=lambda r: (order.get(r["Experiment"], 99), r["Model_Key"]))

    cols = ["Model", "Training_Years", "Test_Year", "Experiment", "Accuracy",
            "Precision_Macro", "Recall_Macro", "F1_Macro", "Kappa",
            "Training_Time_s", "Inference_Time_s", "Model_Size_Bytes", "Params",
            "Input_Type", "Validation_Type", "Checkpoint_Path", "N_Test",
            "Model_Key", "Architecture", "Note"]
    import csv
    csv_path = os.path.join(RESULTS, "model_comparison.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k))
                        for k in cols})

    pc_cols = ["Experiment", "Model_Key", "Class_ID", "Class_Name_Out",
               "Class_Name_Native", "Precision", "Recall", "F1"]
    pc_path = os.path.join(RESULTS, "per_class_metrics.csv")
    with open(pc_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pc_cols)
        w.writeheader()
        w.writerows(per_class)

    _markdown(rows, os.path.join(RESULTS, "model_comparison.md"))
    _environment()
    _recommendation(rows, os.path.join(RESULTS, "recommendation.json"))
    print(f"wrote {csv_path} ({len(rows)} rows), {pc_path}, "
          f"model_comparison.md, environment.json, recommendation.json")


def _per_class(store, row, m):
    out_names = {"1": "Built-up", "2": "Vegetation", "3": "Water", "4": "Open Land"}
    native = {"1": "Water", "2": "Urban", "3": "Soil", "4": "Vegetation"}
    for cid, v in m.get("per_class", {}).items():
        store.append({"Experiment": row["Experiment"], "Model_Key": row["Model_Key"],
                      "Class_ID": cid, "Class_Name_Out": out_names[cid],
                      "Class_Name_Native": native[cid],
                      "Precision": round(v["precision"], 4),
                      "Recall": round(v["recall"], 4),
                      "F1": round(v["f1"], 4)})


def _markdown(rows, path):
    cols = ["Model", "Training_Years", "Test_Year", "Experiment", "Accuracy",
            "Precision_Macro", "Recall_Macro", "F1_Macro", "Kappa",
            "Training_Time_s", "Inference_Time_s", "Input_Type",
            "Validation_Type", "Checkpoint_Path"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _environment():
    import numpy
    import pandas
    import rasterio
    import sklearn
    import torch
    import xgboost
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": "none (CPU-only host)",
        "numpy": numpy.__version__, "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "torch": torch.__version__, "rasterio": rasterio.__version__,
        "torch_mkldnn_disabled": not torch.backends.mkldnn.enabled,
        "seed_everywhere": C.SEED,
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    C.save_json(env, os.path.join(RESULTS, "environment.json"))


def _recommendation(rows, path):
    """Mean F1 per model over the three temporal experiments (E1,E2,E3)."""
    temporal = [r for r in rows if r["Experiment"].startswith(("E1", "E2", "E3"))]
    sameyear = [r for r in rows if r["Experiment"].startswith("E0")]
    agg = {}
    for r in rows:
        k = r["Model_Key"]
        agg.setdefault(k, {"temporal_f1": [], "sameyear_f1": [],
                           "temporal_acc": [], "train_time_s": []})
    for r in temporal:
        agg[r["Model_Key"]]["temporal_f1"].append(r["F1_Macro"])
        agg[r["Model_Key"]]["temporal_acc"].append(r["Accuracy"])
    for r in sameyear:
        agg[r["Model_Key"]]["sameyear_f1"].append(r["F1_Macro"])
    for r in rows:
        agg[r["Model_Key"]]["train_time_s"].append(r["Training_Time_s"])
    out = {}
    for k, v in agg.items():
        out[k] = {
            "mean_temporal_f1_macro": round(sum(v["temporal_f1"]) /
                                            max(len(v["temporal_f1"]), 1), 4),
            "mean_temporal_accuracy": round(sum(v["temporal_acc"]) /
                                            max(len(v["temporal_acc"]), 1), 4),
            "mean_sameyear_f1_macro": round(sum(v["sameyear_f1"]) /
                                            max(len(v["sameyear_f1"]), 1), 4),
            "mean_train_time_s": round(sum(v["train_time_s"]) /
                                       max(len(v["train_time_s"]), 1), 1),
            "n_temporal_runs": len(v["temporal_f1"]),
        }
    # chosen model: highest mean macro-F1 over the three temporal experiments
    # (the design document's primary criterion); wins_per_experiment shows how
    # consistently it leads on every single run.
    ranked = sorted(out, key=lambda k: out[k]["mean_temporal_f1_macro"],
                    reverse=True)
    best = ranked[0]
    wins = {}
    for exp in sorted({r["Experiment"] for r in rows}):
        er = [r for r in rows if r["Experiment"] == exp]
        top = max(er, key=lambda r: r["F1_Macro"])["Model_Key"]
        wins[top] = wins.get(top, 0) + 1
    runner = ranked[1] if len(ranked) > 1 else None
    recommendation = {
        "aggregation": "unweighted mean over experiments, seed 42",
        "by_model": out,
        "chosen_model": best,
        "criterion": "highest mean_temporal_f1_macro over E1/E2/E3",
        "margin_over_runner_up": round(out[best]["mean_temporal_f1_macro"] -
                                        out[runner]["mean_temporal_f1_macro"], 4)
        if runner else None,
        "experiments_won_by_model": wins,
        "n_experiments_total": sum(wins.values()),
        "caveats": [
            "adapted architectures, not the published MTS/Bansal-Tripathi models",
            "single seed (42); label provenance unverified",
            "random-split metrics optimistic vs blocked spatial split "
            "(see results/spatial_validation.*)",
        ],
    }
    C.save_json(recommendation, path)
    print(json.dumps(out, indent=2))
    print(f"chosen_model: {best} "
          f"(mean temporal F1 {out[best]['mean_temporal_f1_macro']})")


if __name__ == "__main__":
    main()
