import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


TARGETS = ["at_least_one", "at_least_two", "at_least_three"]


def parse_int_list(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    nums = re.findall(r"-?\d+", str(s))
    return [int(x) for x in nums]


def get_publishers_universe(stage1_dir: Path):
    ht = pd.read_parquet(stage1_dir / "history_train.parquet", columns=["publisher"])
    pubs = np.sort(ht["publisher"].astype(int).unique())
    return pubs.astype(int).tolist()


def hod_counts_from_start_len(hod_start: np.ndarray, length: np.ndarray) -> np.ndarray:
    n = len(hod_start)
    out = np.zeros((n, 24), dtype=np.float32)

    full = (length // 24).astype(np.int64)
    rem = (length % 24).astype(np.int64)

    out += full[:, None].astype(np.float32)

    for i in range(23):
        idx = np.where(rem > i)[0]
        if idx.size == 0:
            break
        h = (hod_start[idx] + i) % 24
        out[idx, h] += 1.0

    return out


def build_features(df: pd.DataFrame, pub_universe: list[int]) -> pd.DataFrame:
    hour_start = pd.to_numeric(df["hour_start"], errors="coerce").astype(np.int64).to_numpy()
    hour_end = pd.to_numeric(df["hour_end"], errors="coerce").astype(np.int64).to_numpy()
    window_len = (hour_end - hour_start + 1).clip(min=1)

    hod_start = (hour_start % 24).astype(np.int64)
    hod_end = (hour_end % 24).astype(np.int64)

    X = pd.DataFrame(index=df.index)
    X["cpm"] = pd.to_numeric(df["cpm"], errors="coerce").astype(np.float64)
    X["log_cpm"] = np.log1p(X["cpm"].clip(lower=0.0))
    X["audience_size"] = pd.to_numeric(df["audience_size"], errors="coerce").astype(np.float64)
    X["window_length"] = window_len.astype(np.float64)
    X["hod_start"] = hod_start.astype(np.float64)
    X["hod_end"] = hod_end.astype(np.float64)
    X["cpm_per_hour"] = (X["cpm"] / X["window_length"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X["cpm_x_window"] = X["log_cpm"] * np.log1p(X["window_length"])

    pubs_raw = df["publishers"].astype("string").fillna("")
    X["n_publishers"] = pubs_raw.apply(lambda s: 0 if s == "" else len(str(s).split(","))).astype(np.float64)

    hod_mat = hod_counts_from_start_len(hod_start, window_len.astype(np.int64))
    hod_mat = hod_mat / window_len.reshape(-1, 1).astype(np.float32)
    for h in range(24):
        X[f"hod_frac_{h}"] = hod_mat[:, h].astype(np.float32)

    pub_to_col = {p: f"pub_{p}" for p in pub_universe}
    for p in pub_universe:
        X[pub_to_col[p]] = 0.0

    for i, s in enumerate(pubs_raw.tolist()):
        lst = parse_int_list(s)
        if not lst:
            continue
        for p in set(lst):
            col = pub_to_col.get(int(p))
            if col is not None:
                X.iat[i, X.columns.get_loc(col)] = 1.0

    return X


def clip_monotone(p1, p2, p3):
    p1 = np.clip(p1, 0.0, 1.0)
    p2 = np.clip(p2, 0.0, 1.0)
    p3 = np.clip(p3, 0.0, 1.0)
    p2 = np.minimum(p2, p1)
    p3 = np.minimum(p3, p2)
    return p1, p2, p3


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", type=str, default="artifacts/stage1")
    ap.add_argument("--stage2-dir", type=str, default="artifacts/stage2_full")
    ap.add_argument("--models-out-dir", type=str, default="artifacts/stage7")
    ap.add_argument("--pred-out-path", type=str, default="artifacts/stage7/predictions_v2.tsv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=3000)
    args = ap.parse_args()

    stage1_dir = Path(args.stage1_dir)
    stage2_dir = Path(args.stage2_dir)
    models_out_dir = Path(args.models_out_dir)
    models_out_dir.mkdir(parents=True, exist_ok=True)

    pub_universe = get_publishers_universe(stage1_dir)

    campaigns = pd.read_parquet(stage2_dir / "offline_campaigns.parquet")
    answers = pd.read_parquet(stage2_dir / "offline_answers.parquet")
    df = campaigns.merge(answers, on="campaign_id", how="inner")

    X = build_features(df, pub_universe)
    y = {t: df[t].astype(np.float64).to_numpy() for t in TARGETS}

    X_train, X_val, idx_train, idx_val = train_test_split(
        X, np.arange(len(X)), test_size=0.2, random_state=args.seed
    )

    metrics_offline = {}
    for t in TARGETS:
        model = CatBoostRegressor(
            iterations=args.iters,
            learning_rate=0.03,
            depth=8,
            loss_function="MAE",
            random_seed=args.seed,
            verbose=200,
        )
        model.fit(
            X_train, y[t][idx_train],
            eval_set=(X_val, y[t][idx_val]),
            early_stopping_rounds=150,
            use_best_model=True,
        )
        model_path = models_out_dir / f"cb_v2_{t}.cbm"
        model.save_model(str(model_path))

        pred = model.predict(X_val).astype(np.float64)
        pred = np.clip(pred, 0.0, 1.0)
        metrics_offline[t] = {
            "MAE": float(mean_absolute_error(y[t][idx_val], pred)),
            "RMSE": rmse(y[t][idx_val], pred),
            "R2": float(r2_score(y[t][idx_val], pred)),
            "best_iteration": int(model.get_best_iteration() or 0),
        }

    validate = pd.read_csv(Path("data") / "validate.tsv", sep="\t")
    Xv = build_features(validate, pub_universe)

    preds = {}
    for t in TARGETS:
        m = CatBoostRegressor()
        m.load_model(str(models_out_dir / f"cb_v2_{t}.cbm"))
        preds[t] = m.predict(Xv).astype(np.float64)

    p1, p2, p3 = clip_monotone(preds["at_least_one"], preds["at_least_two"], preds["at_least_three"])
    out = pd.DataFrame({"at_least_one": p1, "at_least_two": p2, "at_least_three": p3})
    out_path = Path(args.pred_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)

    answers_path = Path("data") / "validate_answers.tsv"
    metrics_validate = None
    if answers_path.exists():
        ans = pd.read_csv(answers_path, sep="\t")
        metrics_validate = {}
        for t in TARGETS:
            yt = ans[t].astype(np.float64).to_numpy()
            yp = out[t].astype(np.float64).to_numpy()
            metrics_validate[t] = {
                "MAE": float(mean_absolute_error(yt, yp)),
                "RMSE": rmse(yt, yp),
                "R2": float(r2_score(yt, yp)),
            }
        metrics_validate["overall"] = {
            "mean_MAE": float(np.mean([metrics_validate[t]["MAE"] for t in TARGETS])),
            "mean_RMSE": float(np.mean([metrics_validate[t]["RMSE"] for t in TARGETS])),
            "mean_R2": float(np.mean([metrics_validate[t]["R2"] for t in TARGETS])),
        }

    report = {
        "pub_universe_size": int(len(pub_universe)),
        "features_n": int(X.shape[1]),
        "offline_metrics": metrics_offline,
        "validate_metrics": metrics_validate,
        "pred_out_path": str(out_path),
    }

    with open(models_out_dir / "report_stage7.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report.get("validate_metrics", {}), ensure_ascii=False, indent=2))
    print(f"Wrote: {models_out_dir / 'report_stage7.json'}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
