import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = ["at_least_one", "at_least_two", "at_least_three"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def extract_campaign_features(validate_df: pd.DataFrame) -> pd.DataFrame:
    df = validate_df.copy()
    out = pd.DataFrame(index=df.index)
    out["cpm"] = df["cpm"].astype(float)
    out["audience_size"] = pd.to_numeric(df["audience_size"], errors="coerce").astype(float)
    out["window_length"] = (pd.to_numeric(df["hour_end"], errors="coerce") - pd.to_numeric(df["hour_start"], errors="coerce") + 1).astype(float)
    out["hod_start"] = (pd.to_numeric(df["hour_start"], errors="coerce") % 24).astype(float)
    out["hod_end"] = (pd.to_numeric(df["hour_end"], errors="coerce") % 24).astype(float)
    out["log_cpm"] = np.log1p(out["cpm"].clip(lower=0.0))
    out["cpm_per_hour"] = out["cpm"] / out["window_length"].replace(0.0, np.nan)
    out["cpm_per_hour"] = out["cpm_per_hour"].fillna(0.0)

    pubs = df["publishers"].astype("string").fillna("")
    out["n_publishers"] = pubs.apply(lambda s: 0 if s == "" else len(str(s).split(","))).astype(float)
    return out


def bin_by_quantiles(x: pd.Series, q=(0.2, 0.4, 0.6, 0.8)):
    qs = x.quantile(list(q)).to_numpy()
    edges = [-np.inf] + qs.tolist() + [np.inf]
    labels = [f"q{i}" for i in range(len(edges) - 1)]
    return pd.cut(x, bins=edges, labels=labels, include_lowest=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--pred-path", type=str, default="artifacts/stage4/predictions.tsv")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage6")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pred_path = Path(args.pred_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_path = data_dir / "validate.tsv"
    answers_path = data_dir / "validate_answers.tsv"

    validate = pd.read_csv(validate_path, sep="\t")
    answers = pd.read_csv(answers_path, sep="\t")
    preds = pd.read_csv(pred_path, sep="\t")

    for t in TARGETS:
        if t not in preds.columns:
            raise ValueError(f"predictions missing column: {t}")
        if t not in answers.columns:
            raise ValueError(f"validate_answers missing column: {t}")

    if len(preds) != len(answers):
        raise ValueError(f"Row count mismatch: preds={len(preds)} answers={len(answers)}")

    preds_clipped = preds.copy()
    for t in TARGETS:
        preds_clipped[t] = preds_clipped[t].astype(float).clip(0.0, 1.0)

    y_true = {t: answers[t].astype(float).to_numpy() for t in TARGETS}
    y_pred = {t: preds_clipped[t].astype(float).to_numpy() for t in TARGETS}

    report = {"overall": {}, "by_target": {}, "sanity": {}, "slices": {}}

    mono_12 = float(np.mean(preds_clipped["at_least_one"] >= preds_clipped["at_least_two"]))
    mono_23 = float(np.mean(preds_clipped["at_least_two"] >= preds_clipped["at_least_three"]))
    mono_13 = float(np.mean(preds_clipped["at_least_one"] >= preds_clipped["at_least_three"]))
    report["sanity"]["monotonic_share_p1_ge_p2"] = mono_12
    report["sanity"]["monotonic_share_p2_ge_p3"] = mono_23
    report["sanity"]["monotonic_share_p1_ge_p3"] = mono_13
    report["sanity"]["any_out_of_range_before_clip"] = bool(((preds[TARGETS] < 0) | (preds[TARGETS] > 1)).any().any())

    maes = []
    rmses = []
    r2s = []

    per_row = pd.DataFrame(index=np.arange(len(preds_clipped)))
    for t in TARGETS:
        mae = float(mean_absolute_error(y_true[t], y_pred[t]))
        r = rmse(y_true[t], y_pred[t])
        r2 = float(r2_score(y_true[t], y_pred[t]))

        report["by_target"][t] = {"MAE": mae, "RMSE": r, "R2": r2}
        maes.append(mae)
        rmses.append(r)
        r2s.append(r2)

        per_row[f"err_abs_{t}"] = np.abs(y_true[t] - y_pred[t])
        per_row[f"y_true_{t}"] = y_true[t]
        per_row[f"y_pred_{t}"] = y_pred[t]

    report["overall"]["mean_MAE"] = float(np.mean(maes))
    report["overall"]["mean_RMSE"] = float(np.mean(rmses))
    report["overall"]["mean_R2"] = float(np.mean(r2s))

    camp_feat = extract_campaign_features(validate)
    per_row = pd.concat([camp_feat, per_row], axis=1)
    per_row["err_abs_mean3"] = per_row[[f"err_abs_{t}" for t in TARGETS]].mean(axis=1)

    slice_cols = ["window_length", "cpm", "n_publishers", "audience_size", "cpm_per_hour"]
    for col in slice_cols:
        b = bin_by_quantiles(per_row[col].replace([np.inf, -np.inf], np.nan).fillna(per_row[col].median()))
        grp = per_row.groupby(b, observed=True)["err_abs_mean3"].agg(["count", "mean", "median"]).reset_index()
        report["slices"][col] = grp.to_dict(orient="records")

    per_row_out = out_dir / "per_row_errors.tsv"
    per_row.to_csv(per_row_out, sep="\t", index=False)

    report_out = out_dir / "report_validate.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {per_row_out}")
    print(f"Wrote: {report_out}")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
