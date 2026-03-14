import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_div(a, b):
    b = np.where(b == 0, np.nan, b)
    out = a / b
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", type=str, default="artifacts/stage1")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage9")
    args = ap.parse_args()

    stage1_dir = Path(args.stage1_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    users = pd.read_parquet(stage1_dir / "users.parquet")
    ht = pd.read_parquet(stage1_dir / "history_train.parquet")
    st = pd.read_parquet(stage1_dir / "sessions_train.parquet")

    users["user_id"] = users["user_id"].astype(np.int64)
    ht["user_id"] = ht["user_id"].astype(np.int64)
    ht["publisher"] = ht["publisher"].astype(np.int32)
    ht["hod"] = ht["hod"].astype(np.int16)
    ht["day"] = ht["day"].astype(np.int64)
    ht["cpm"] = ht["cpm"].astype(np.float64)

    st["user_id"] = st["user_id"].astype(np.int64)
    st["impressions"] = st["impressions"].astype(np.int64)
    st["session_len_hours"] = st["session_len_hours"].astype(np.int64)

    pub_universe = np.sort(ht["publisher"].unique()).astype(np.int32)
    hod_universe = np.arange(24, dtype=np.int16)

    base = users[["user_id", "sex", "age", "city_id"]].copy()
    base["sex"] = base["sex"].astype(np.int16)
    base["age"] = base["age"].astype(np.int16)
    base["city_id"] = base["city_id"].astype(np.int32)

    base["age_is_missing"] = (base["age"] == 0).astype(np.int8)
    base["city_is_missing"] = (base["city_id"] == 0).astype(np.int8)
    base["sex_is_missing"] = (base["sex"] == 0).astype(np.int8)

    g = ht.groupby("user_id", sort=False)
    agg = g.agg(
        impressions_total=("hour", "size"),
        days_active=("day", "nunique"),
        publishers_nunique=("publisher", "nunique"),
        cpm_mean=("cpm", "mean"),
        cpm_std=("cpm", "std"),
        cpm_min=("cpm", "min"),
        cpm_p50=("cpm", lambda x: float(np.quantile(x.to_numpy(), 0.50))),
        cpm_p90=("cpm", lambda x: float(np.quantile(x.to_numpy(), 0.90))),
        cpm_max=("cpm", "max"),
        hod_nunique=("hod", "nunique"),
    ).reset_index()

    agg["cpm_std"] = agg["cpm_std"].fillna(0.0)
    agg["impr_per_day_active"] = safe_div(agg["impressions_total"].to_numpy(), agg["days_active"].to_numpy())

    stg = st.groupby("user_id", sort=False).agg(
        sessions_total=("session_id", "nunique"),
        session_len_mean=("session_len_hours", "mean"),
        session_len_p90=("session_len_hours", lambda x: float(np.quantile(x.to_numpy(), 0.90))),
        impressions_per_session_mean=("impressions", "mean"),
        impressions_per_session_p90=("impressions", lambda x: float(np.quantile(x.to_numpy(), 0.90))),
    ).reset_index()
    stg["session_len_mean"] = stg["session_len_mean"].fillna(0.0)
    stg["impressions_per_session_mean"] = stg["impressions_per_session_mean"].fillna(0.0)

    hod_cnt = (
        ht.groupby(["user_id", "hod"], sort=False)
        .size()
        .rename("cnt")
        .reset_index()
        .pivot(index="user_id", columns="hod", values="cnt")
        .reindex(columns=hod_universe, fill_value=0)
    )
    hod_cnt.columns = [f"hod_cnt_{int(c)}" for c in hod_cnt.columns]
    hod_cnt = hod_cnt.reset_index()

    hod_frac = hod_cnt.copy()
    cnt_cols = [c for c in hod_frac.columns if c.startswith("hod_cnt_")]
    row_sum = hod_frac[cnt_cols].sum(axis=1).to_numpy()
    for c in cnt_cols:
        hod_frac[c.replace("hod_cnt_", "hod_frac_")] = safe_div(hod_frac[c].to_numpy(), row_sum)
    hod_frac = hod_frac[["user_id"] + [c for c in hod_frac.columns if c.startswith("hod_frac_")]]

    pub_cnt = (
        ht.groupby(["user_id", "publisher"], sort=False)
        .size()
        .rename("cnt")
        .reset_index()
        .pivot(index="user_id", columns="publisher", values="cnt")
        .reindex(columns=pub_universe, fill_value=0)
    )
    pub_cnt.columns = [f"pub_cnt_{int(c)}" for c in pub_cnt.columns]
    pub_cnt = pub_cnt.reset_index()

    pub_frac = pub_cnt.copy()
    pub_cols = [c for c in pub_frac.columns if c.startswith("pub_cnt_")]
    row_sum_p = pub_frac[pub_cols].sum(axis=1).to_numpy()
    for c in pub_cols:
        pub_frac[c.replace("pub_cnt_", "pub_frac_")] = safe_div(pub_frac[c].to_numpy(), row_sum_p)
    pub_frac = pub_frac[["user_id"] + [c for c in pub_frac.columns if c.startswith("pub_frac_")]]

    df = base.merge(agg, on="user_id", how="left").merge(stg, on="user_id", how="left")
    df = df.merge(hod_frac, on="user_id", how="left").merge(pub_frac, on="user_id", how="left")

    fill_zero_cols = [
        "impressions_total", "days_active", "publishers_nunique", "cpm_mean", "cpm_std", "cpm_min",
        "cpm_p50", "cpm_p90", "cpm_max", "hod_nunique", "impr_per_day_active",
        "sessions_total", "session_len_mean", "session_len_p90", "impressions_per_session_mean",
        "impressions_per_session_p90",
    ]
    for c in fill_zero_cols:
        df[c] = df[c].fillna(0)

    for c in [c for c in df.columns if c.startswith("hod_frac_") or c.startswith("pub_frac_")]:
        df[c] = df[c].fillna(0.0)

    df["log_impressions_total"] = np.log1p(df["impressions_total"].astype(np.float64))
    df["log_sessions_total"] = np.log1p(df["sessions_total"].astype(np.float64))
    df["log_days_active"] = np.log1p(df["days_active"].astype(np.float64))

    feature_cols = [c for c in df.columns if c != "user_id"]
    df = df[["user_id"] + feature_cols]

    out_path = out_dir / "user_features.parquet"
    df.to_parquet(out_path, index=False)

    report = {
        "users_total": int(users["user_id"].nunique()),
        "history_train_rows": int(len(ht)),
        "sessions_train_rows": int(len(st)),
        "pub_universe_size": int(len(pub_universe)),
        "feature_dim": int(len(feature_cols)),
        "users_with_history_share": float(ht["user_id"].nunique() / users["user_id"].nunique()),
        "output_path": str(out_path),
    }
    with open(out_dir / "report_stage9_1.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_dir / 'report_stage9_1.json'}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
