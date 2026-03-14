import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_USERS_COLS = ["user_id", "sex", "age", "city_id"]
REQUIRED_HISTORY_COLS = ["hour", "cpm", "publisher", "user_id"]
REQUIRED_VALIDATE_COLS = ["cpm", "hour_start", "hour_end", "publishers", "audience_size", "user_ids"]


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def ensure_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")


def coerce_int(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(s, errors="raise")
    if (out % 1 != 0).any():
        raise ValueError(f"{name}: expected integer-like values")
    return out.astype(np.int64)


def build_sessions(history: pd.DataFrame, gap_hours: int = 6) -> pd.DataFrame:
    h = history.sort_values(["user_id", "hour"]).reset_index(drop=True)
    prev_hour = h.groupby("user_id")["hour"].shift(1)
    new_sess = prev_hour.isna() | ((h["hour"] - prev_hour) > gap_hours)
    h["session_idx"] = new_sess.groupby(h["user_id"]).cumsum().astype(np.int64)
    h["session_id"] = (h["user_id"].astype(str) + "_" + h["session_idx"].astype(str)).astype("string")
    return h


def sessions_table(history_with_sessions: pd.DataFrame) -> pd.DataFrame:
    g = history_with_sessions.groupby(["user_id", "session_id"], sort=False)
    sess = g.agg(
        session_start_hour=("hour", "min"),
        session_end_hour=("hour", "max"),
        impressions=("hour", "size"),
        publishers_nunique=("publisher", "nunique"),
        cpm_mean=("cpm", "mean"),
        cpm_max=("cpm", "max"),
    ).reset_index()
    sess["session_len_hours"] = (sess["session_end_hour"] - sess["session_start_hour"]).astype(np.int64)
    sess["day_start"] = (sess["session_start_hour"] // 24).astype(np.int64)
    sess["hod_start"] = (sess["session_start_hour"] % 24).astype(np.int64)
    return sess


def split_by_days(history: pd.DataFrame):
    days = np.sort(history["day"].unique())
    if len(days) < 2:
        raise ValueError("history: not enough distinct days to split")
    split_point = len(days) // 2
    train_days = set(days[:split_point])
    valid_days = set(days[split_point:])
    train = history[history["day"].isin(train_days)].copy()
    valid = history[history["day"].isin(valid_days)].copy()
    split_info = {
        "n_days_total": int(len(days)),
        "split_point_index": int(split_point),
        "train_day_min": int(min(train_days)),
        "train_day_max": int(max(train_days)),
        "valid_day_min": int(min(valid_days)),
        "valid_day_max": int(max(valid_days)),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
    }
    return train, valid, split_info


def describe_df(df: pd.DataFrame, name: str) -> dict:
    out = {"name": name, "rows": int(len(df)), "cols": int(df.shape[1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=".")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage1")
    ap.add_argument("--session-gap-hours", type=int, default=6)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    users_path = data_dir / "users.tsv"
    history_path = data_dir / "history.tsv"
    validate_path = data_dir / "validate.tsv"
    validate_answers_path = data_dir / "validate_answers.tsv"

    users = read_tsv(users_path)
    history = read_tsv(history_path)
    validate = read_tsv(validate_path)
    validate_answers = read_tsv(validate_answers_path) if validate_answers_path.exists() else None

    ensure_cols(users, REQUIRED_USERS_COLS, "users.tsv")
    ensure_cols(history, REQUIRED_HISTORY_COLS, "history.tsv")
    ensure_cols(validate, REQUIRED_VALIDATE_COLS, "validate.tsv")

    users = users[REQUIRED_USERS_COLS].copy()
    history = history[REQUIRED_HISTORY_COLS].copy()
    validate = validate[REQUIRED_VALIDATE_COLS].copy()

    users["user_id"] = coerce_int(users["user_id"], "users.user_id")
    users["sex"] = coerce_int(users["sex"], "users.sex").astype(np.int16)
    users["age"] = coerce_int(users["age"], "users.age").astype(np.int16)
    users["city_id"] = coerce_int(users["city_id"], "users.city_id").astype(np.int32)

    history["user_id"] = coerce_int(history["user_id"], "history.user_id")
    history["hour"] = coerce_int(history["hour"], "history.hour")
    history["publisher"] = coerce_int(history["publisher"], "history.publisher").astype(np.int32)
    history["cpm"] = pd.to_numeric(history["cpm"], errors="raise").astype(np.float64)

    history["day"] = (history["hour"] // 24).astype(np.int64)
    history["hod"] = (history["hour"] % 24).astype(np.int16)

    validate["hour_start"] = coerce_int(validate["hour_start"], "validate.hour_start")
    validate["hour_end"] = coerce_int(validate["hour_end"], "validate.hour_end")
    validate["cpm"] = pd.to_numeric(validate["cpm"], errors="raise").astype(np.float64)
    validate["audience_size"] = coerce_int(validate["audience_size"], "validate.audience_size").astype(np.int64)
    validate["publishers"] = validate["publishers"].astype("string")
    validate["user_ids"] = validate["user_ids"].astype("string")

    history_ws = build_sessions(history, gap_hours=args.session_gap_hours)
    sessions = sessions_table(history_ws)

    train_hist, valid_hist, split_info = split_by_days(history_ws)
    train_sessions = sessions_table(train_hist)
    valid_sessions = sessions_table(valid_hist)

    report = {}
    report["files"] = [
        describe_df(users, "users"),
        describe_df(history, "history_raw"),
        describe_df(history_ws, "history_with_sessions"),
        describe_df(sessions, "sessions_full"),
        describe_df(train_hist, "history_train"),
        describe_df(valid_hist, "history_valid"),
        describe_df(train_sessions, "sessions_train"),
        describe_df(valid_sessions, "sessions_valid"),
        describe_df(validate, "validate"),
    ]
    report["split"] = split_info
    report["users"] = {
        "n_users": int(users["user_id"].nunique()),
        "age_zero_share": float((users["age"] == 0).mean()),
        "city_zero_share": float((users["city_id"] == 0).mean()),
        "sex_zero_share": float((users["sex"] == 0).mean()),
    }
    report["history"] = {
        "n_users": int(history_ws["user_id"].nunique()),
        "n_publishers": int(history_ws["publisher"].nunique()),
        "hour_min": int(history_ws["hour"].min()),
        "hour_max": int(history_ws["hour"].max()),
        "days_nunique": int(history_ws["day"].nunique()),
        "cpm_min": float(history_ws["cpm"].min()),
        "cpm_p50": float(history_ws["cpm"].quantile(0.50)),
        "cpm_p90": float(history_ws["cpm"].quantile(0.90)),
        "cpm_max": float(history_ws["cpm"].max()),
        "sessions_nunique": int(history_ws["session_id"].nunique()),
    }
    report["validate"] = {
        "rows": int(len(validate)),
        "audience_size_p50": float(validate["audience_size"].quantile(0.50)),
        "audience_size_p90": float(validate["audience_size"].quantile(0.90)),
        "hour_start_min": int(validate["hour_start"].min()),
        "hour_end_max": int(validate["hour_end"].max()),
    }
    if validate_answers is not None:
        report["validate_answers"] = describe_df(validate_answers, "validate_answers")

    users.to_parquet(out_dir / "users.parquet", index=False)
    history_ws.to_parquet(out_dir / "history_with_sessions.parquet", index=False)
    sessions.to_parquet(out_dir / "sessions_full.parquet", index=False)
    train_hist.to_parquet(out_dir / "history_train.parquet", index=False)
    valid_hist.to_parquet(out_dir / "history_valid.parquet", index=False)
    train_sessions.to_parquet(out_dir / "sessions_train.parquet", index=False)
    valid_sessions.to_parquet(out_dir / "sessions_valid.parquet", index=False)
    validate.to_parquet(out_dir / "validate.parquet", index=False)
    if validate_answers is not None:
        validate_answers.to_parquet(out_dir / "validate_answers.parquet", index=False)

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
