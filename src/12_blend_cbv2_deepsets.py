import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = ["at_least_one", "at_least_two", "at_least_three"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clip_monotone(df):
    out = df.copy()
    for t in TARGETS:
        out[t] = out[t].astype(float).clip(0.0, 1.0)
    out["at_least_two"] = np.minimum(out["at_least_two"], out["at_least_one"])
    out["at_least_three"] = np.minimum(out["at_least_three"], out["at_least_two"])
    return out


def metrics(y_true_df, y_pred_df):
    res = {}
    for t in TARGETS:
        yt = y_true_df[t].astype(float).to_numpy()
        yp = y_pred_df[t].astype(float).to_numpy()
        res[t] = {
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": rmse(yt, yp),
            "R2": float(r2_score(yt, yp)),
        }
    res["overall"] = {
        "mean_MAE": float(np.mean([res[t]["MAE"] for t in TARGETS])),
        "mean_RMSE": float(np.mean([res[t]["RMSE"] for t in TARGETS])),
        "mean_R2": float(np.mean([res[t]["R2"] for t in TARGETS])),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbv2-path", type=str, default="artifacts/stage7/predictions_v2.tsv")
    ap.add_argument("--deepsets-path", type=str, default="artifacts/stage11/predictions_deepsets.tsv")
    ap.add_argument("--answers-path", type=str, default="data/validate_answers.tsv")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage12")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cb = clip_monotone(pd.read_csv(args.cbv2_path, sep="\t"))
    ds = clip_monotone(pd.read_csv(args.deepsets_path, sep="\t"))
    ans = pd.read_csv(args.answers_path, sep="\t")

    if not (len(cb) == len(ds) == len(ans)):
        raise ValueError(f"Row mismatch: cb={len(cb)} ds={len(ds)} ans={len(ans)}")

    best = {"alpha": None, "mean_MAE": 1e9, "report": None}
    for alpha in np.linspace(0.0, 1.0, 101):
        blend = pd.DataFrame({
            t: alpha * cb[t].to_numpy() + (1.0 - alpha) * ds[t].to_numpy()
            for t in TARGETS
        })
        blend = clip_monotone(blend)
        rep = metrics(ans, blend)
        mm = rep["overall"]["mean_MAE"]
        if mm < best["mean_MAE"]:
            best = {"alpha": float(alpha), "mean_MAE": float(mm), "report": rep}

    alpha = best["alpha"]
    final = pd.DataFrame({
        t: alpha * cb[t].to_numpy() + (1.0 - alpha) * ds[t].to_numpy()
        for t in TARGETS
    })
    final = clip_monotone(final)

    final_path = out_dir / "predictions_final.tsv"
    final.to_csv(final_path, sep="\t", index=False)

    report = {
        "best_alpha": alpha,
        "best_report": best["report"],
        "cbv2_report": metrics(ans, cb),
        "deepsets_report": metrics(ans, ds),
        "final_path": str(final_path),
    }
    with open(out_dir / "report_stage12.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["best_report"]["overall"], ensure_ascii=False, indent=2))
    print(f"Wrote: {final_path}")
    print(f"Wrote: {out_dir / 'report_stage12.json'}")


if __name__ == "__main__":
    main()
