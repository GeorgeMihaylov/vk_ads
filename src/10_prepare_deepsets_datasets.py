import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_int_list(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    nums = re.findall(r"-?\d+", str(s))
    return [int(x) for x in nums]


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


def build_campaign_features(df: pd.DataFrame, pub_universe: list[int]) -> pd.DataFrame:
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

    cols = list(X.columns)
    col_to_idx = {c: i for i, c in enumerate(cols)}

    for i, s in enumerate(pubs_raw.tolist()):
        lst = parse_int_list(s)
        if not lst:
            continue
        for p in set(lst):
            col = pub_to_col.get(int(p))
            if col is None:
                continue
            X.iat[i, col_to_idx[col]] = 1.0

    return X


def user_ids_to_indices(sampled_user_ids: np.ndarray, user_id_sorted: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(user_id_sorted, sampled_user_ids)
    ok = (pos >= 0) & (pos < len(user_id_sorted)) & (user_id_sorted[pos] == sampled_user_ids)
    idx = np.where(ok, pos, -1).astype(np.int32)
    return idx


def sample_user_indices_for_row(user_ids_str, K, rng, user_id_sorted):
    u = parse_int_list(user_ids_str)
    if len(u) == 0:
        return np.full((K,), -1, dtype=np.int32)

    u = np.array(u, dtype=np.int64)
    if len(u) >= K:
        sampled = rng.choice(u, size=K, replace=False)
    else:
        pad = np.full((K,), -1, dtype=np.int64)
        pad[: len(u)] = u
        sampled = pad

    idx = np.full((K,), -1, dtype=np.int32)
    mask = sampled != -1
    if mask.any():
        idx[mask] = user_ids_to_indices(sampled[mask], user_id_sorted)
    return idx


def prepare_split(name, df_campaigns, targets_df, K, rng, user_id_sorted, out_dir: Path, pub_universe: list[int]):
    t0 = time.time()

    Xc = build_campaign_features(df_campaigns, pub_universe).to_numpy(np.float32)

    user_idx = np.empty((len(df_campaigns), K), dtype=np.int32)
    user_ids_col = df_campaigns["user_ids"].astype("string").fillna("").to_list()
    for i in range(len(df_campaigns)):
        user_idx[i] = sample_user_indices_for_row(user_ids_col[i], K, rng, user_id_sorted)

    y = None
    if targets_df is not None:
        y = targets_df[["at_least_one", "at_least_two", "at_least_three"]].astype(np.float32).to_numpy()

    np.save(out_dir / f"{name}_campaign_feat.npy", Xc)
    np.save(out_dir / f"{name}_user_idx.npy", user_idx)
    if y is not None:
        np.save(out_dir / f"{name}_targets.npy", y)

    sec = time.time() - t0
    return {
        "name": name,
        "rows": int(len(df_campaigns)),
        "K": int(K),
        "campaign_feat_shape": [int(Xc.shape[0]), int(Xc.shape[1])],
        "user_idx_shape": [int(user_idx.shape[0]), int(user_idx.shape[1])],
        "targets_saved": bool(y is not None),
        "time_sec": float(sec),
        "pad_share": float(np.mean(user_idx == -1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", type=str, default="artifacts/stage1")
    ap.add_argument("--stage2-dir", type=str, default="artifacts/stage2_full")
    ap.add_argument("--stage9-dir", type=str, default="artifacts/stage9")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage10")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stage1_dir = Path(args.stage1_dir)
    stage2_dir = Path(args.stage2_dir)
    stage9_dir = Path(args.stage9_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    uf = pd.read_parquet(stage9_dir / "user_features.parquet")
    uf = uf.sort_values("user_id").reset_index(drop=True)
    user_id_sorted = uf["user_id"].astype(np.int64).to_numpy()
    user_feat = uf.drop(columns=["user_id"]).to_numpy(np.float32)

    np.save(out_dir / "user_id_sorted.npy", user_id_sorted)
    np.save(out_dir / "user_feat.npy", user_feat)

    pub_universe = np.sort(pd.read_parquet(stage1_dir / "history_train.parquet", columns=["publisher"])["publisher"].astype(int).unique()).tolist()

    offline_c = pd.read_parquet(stage2_dir / "offline_campaigns.parquet")
    offline_a = pd.read_parquet(stage2_dir / "offline_answers.parquet").sort_values("campaign_id").reset_index(drop=True)
    offline_c = offline_c.sort_values("campaign_id").reset_index(drop=True)

    validate = pd.read_parquet(stage1_dir / "validate.parquet")
    validate_answers_path = Path("data") / "validate_answers.tsv"
    validate_answers = pd.read_csv(validate_answers_path, sep="\t") if validate_answers_path.exists() else None

    rep = {}
    rep["user_table"] = {
        "users": int(len(user_id_sorted)),
        "feature_dim": int(user_feat.shape[1]),
        "user_id_min": int(user_id_sorted.min()),
        "user_id_max": int(user_id_sorted.max()),
    }
    rep["pub_universe_size"] = int(len(pub_universe))
    rep["K"] = int(args.K)
    rep["seed"] = int(args.seed)

    rep["offline"] = prepare_split(
        "offline",
        offline_c,
        offline_a,
        args.K,
        rng,
        user_id_sorted,
        out_dir,
        pub_universe
    )

    rep["validate"] = prepare_split(
        "validate",
        validate,
        validate_answers,
        args.K,
        rng,
        user_id_sorted,
        out_dir,
        pub_universe
    )

    with open(out_dir / "report_stage10.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_dir / 'report_stage10.json'}")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
