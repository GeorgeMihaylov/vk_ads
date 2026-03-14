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
    txt = str(s)
    nums = re.findall(r"-?\d+", txt)
    return [int(x) for x in nums]


def fmt_int_list(lst):
    return ",".join(str(int(x)) for x in lst)


def build_user_index(history_valid):
    hv = history_valid.sort_values(["user_id", "hour"]).reset_index(drop=True)
    hv["cpm_r"] = np.round(hv["cpm"].astype(np.float64), 2)

    user_ids = hv["user_id"].to_numpy(np.int64)
    hours = hv["hour"].to_numpy(np.int64)
    pubs = hv["publisher"].to_numpy(np.int32)
    cpms = hv["cpm_r"].to_numpy(np.float64)
    sess = hv["session_idx"].to_numpy(np.int64)

    idx = {}
    starts = np.flatnonzero(np.r_[True, user_ids[1:] != user_ids[:-1]])
    ends = np.r_[starts[1:], len(hv)]
    for st, en in zip(starts, ends):
        u = int(user_ids[st])
        idx[u] = (hours[st:en], pubs[st:en], cpms[st:en], sess[st:en])
    return idx


def dp_at_least_k(session_probs):
    dp0, dp1, dp2, dp3p = 1.0, 0.0, 0.0, 0.0
    for p in session_probs:
        p = float(p)
        if p <= 0.0:
            continue
        if p >= 1.0:
            dp3p = dp3p + dp2
            dp2 = dp1
            dp1 = dp0
            dp0 = 0.0
            continue
        new0 = dp0 * (1.0 - p)
        new1 = dp1 * (1.0 - p) + dp0 * p
        new2 = dp2 * (1.0 - p) + dp1 * p
        new3p = dp3p + dp2 * p
        dp0, dp1, dp2, dp3p = new0, new1, new2, new3p
    at_least_1 = 1.0 - dp0
    at_least_2 = 1.0 - (dp0 + dp1)
    at_least_3 = dp3p
    return at_least_1, at_least_2, at_least_3


def user_probs_for_campaign(u, user_index, hour_start, hour_end, publishers_set, cpm_r):
    arr = user_index.get(int(u))
    if arr is None:
        return 0.0, 0.0, 0.0

    hours, pubs, cpms, sess = arr
    left = np.searchsorted(hours, hour_start, side="left")
    right = np.searchsorted(hours, hour_end, side="right")
    if right <= left:
        return 0.0, 0.0, 0.0

    h2 = hours[left:right]
    p2 = pubs[left:right]
    c2 = cpms[left:right]
    s2 = sess[left:right]

    mask = np.isin(p2, list(publishers_set))
    if not mask.any():
        return 0.0, 0.0, 0.0

    p2 = p2[mask]
    c2 = c2[mask]
    s2 = s2[mask]

    session_state = {}
    for si, ci in zip(s2, c2):
        si = int(si)
        st = session_state.get(si)
        if st is None:
            st = [False, 0]
            session_state[si] = st
        if ci < cpm_r:
            st[0] = True
        elif ci == cpm_r:
            st[1] += 1

    probs = []
    for has_lower, m_eq in session_state.values():
        if has_lower:
            probs.append(1.0)
        elif m_eq > 0:
            probs.append(1.0 - (0.5 ** int(m_eq)))
        else:
            probs.append(0.0)

    return dp_at_least_k(probs)


def sample_distributions_from_validate(validate_df):
    pubs_lists = validate_df["publishers"].astype("string").apply(parse_int_list)
    lens = pubs_lists.apply(len).to_numpy(np.int64)
    lens = lens[lens > 0]
    if len(lens) == 0:
        lens = np.array([1], dtype=np.int64)

    unique_lens, counts = np.unique(lens, return_counts=True)
    p_len = counts / counts.sum()

    flat_pubs = []
    for lst in pubs_lists.tolist():
        flat_pubs.extend(lst)
    if len(flat_pubs) == 0:
        pub_ids = np.array([0], dtype=np.int32)
        pub_p = np.array([1.0], dtype=np.float64)
    else:
        pub_ids, pub_counts = np.unique(np.array(flat_pubs, dtype=np.int32), return_counts=True)
        pub_p = pub_counts / pub_counts.sum()

    deltas = (validate_df["hour_end"].astype(np.int64) - validate_df["hour_start"].astype(np.int64)).to_numpy(np.int64)
    deltas = deltas[deltas >= 0]
    if len(deltas) == 0:
        deltas = np.array([23], dtype=np.int64)

    cpms = validate_df["cpm"].astype(np.float64).to_numpy()
    cpms = cpms[np.isfinite(cpms)]
    if len(cpms) == 0:
        cpms = np.array([100.0], dtype=np.float64)

    return {
        "len_values": unique_lens.astype(np.int64),
        "len_probs": p_len.astype(np.float64),
        "pub_ids": pub_ids.astype(np.int32),
        "pub_probs": pub_p.astype(np.float64),
        "deltas": deltas.astype(np.int64),
        "cpms": cpms.astype(np.float64),
    }


def parse_audience_config(s):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    sizes = []
    probs = []
    for p in parts:
        a, b = p.split(":")
        sizes.append(int(a))
        probs.append(float(b))
    probs = np.array(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return np.array(sizes, dtype=np.int64), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", type=str, default="artifacts/stage1")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage2")
    ap.add_argument("--n-campaigns", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-window-hours", type=int, default=168)
    ap.add_argument("--audience-config", type=str, default="512:0.7,1024:0.2,2048:0.1")
    args = ap.parse_args()

    stage1_dir = Path(args.stage1_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hv_path = stage1_dir / "history_valid.parquet"
    val_path = stage1_dir / "validate.parquet"

    history_valid = pd.read_parquet(hv_path)
    validate = pd.read_parquet(val_path)

    required_cols = {"hour", "cpm", "publisher", "user_id", "session_idx"}
    missing = sorted(list(required_cols - set(history_valid.columns)))
    if missing:
        raise ValueError(f"history_valid missing columns: {missing}")

    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    user_index = build_user_index(history_valid)
    t_index = time.time() - t0

    active_users = np.array(sorted(user_index.keys()), dtype=np.int64)
    hv_hour_min = int(history_valid["hour"].min())
    hv_hour_max = int(history_valid["hour"].max())

    dist = sample_distributions_from_validate(validate)
    aud_sizes, aud_probs = parse_audience_config(args.audience_config)

    campaigns = []
    answers = []

    t1 = time.time()
    for cid in range(int(args.n_campaigns)):
        aud_size = int(rng.choice(aud_sizes, p=aud_probs))
        if aud_size > len(active_users):
            aud_size = int(len(active_users))

        users = rng.choice(active_users, size=aud_size, replace=False)

        n_pubs = int(rng.choice(dist["len_values"], p=dist["len_probs"]))
        n_pubs = max(1, min(n_pubs, len(dist["pub_ids"])))
        pubs = rng.choice(dist["pub_ids"], size=n_pubs, replace=False, p=dist["pub_probs"])
        pubs = np.unique(pubs.astype(np.int32))
        pubs_set = set(int(x) for x in pubs.tolist())

        delta = int(rng.choice(dist["deltas"]))
        win_len = int(min(max(delta, 0) + 1, args.max_window_hours))
        if hv_hour_max - hv_hour_min + 1 <= win_len:
            hour_start = hv_hour_min
            hour_end = hv_hour_max
        else:
            hour_start = int(rng.integers(hv_hour_min, hv_hour_max - win_len + 2))
            hour_end = int(hour_start + win_len - 1)

        cpm = float(rng.choice(dist["cpms"]))
        cpm_r = float(np.round(cpm, 2))

        p1s = []
        p2s = []
        p3s = []
        for u in users:
            a1, a2, a3 = user_probs_for_campaign(int(u), user_index, hour_start, hour_end, pubs_set, cpm_r)
            p1s.append(a1)
            p2s.append(a2)
            p3s.append(a3)

        at1 = float(np.mean(p1s)) if len(p1s) else 0.0
        at2 = float(np.mean(p2s)) if len(p2s) else 0.0
        at3 = float(np.mean(p3s)) if len(p3s) else 0.0

        campaigns.append(
            {
                "campaign_id": int(cid),
                "cpm": float(cpm),
                "hour_start": int(hour_start),
                "hour_end": int(hour_end),
                "publishers": fmt_int_list(pubs.tolist()),
                "audience_size": int(len(users)),
                "user_ids": fmt_int_list(users.tolist()),
            }
        )
        answers.append(
            {
                "campaign_id": int(cid),
                "at_least_one": at1,
                "at_least_two": at2,
                "at_least_three": at3,
            }
        )

    gen_time = time.time() - t1

    campaigns_df = pd.DataFrame(campaigns)
    answers_df = pd.DataFrame(answers)

    campaigns_df.to_parquet(out_dir / "offline_campaigns.parquet", index=False)
    answers_df.to_parquet(out_dir / "offline_answers.parquet", index=False)

    campaigns_df.to_csv(out_dir / "offline_campaigns.tsv", sep="\t", index=False)
    answers_df.to_csv(out_dir / "offline_answers.tsv", sep="\t", index=False)

    report = {
        "stage1_dir": str(stage1_dir),
        "out_dir": str(out_dir),
        "n_campaigns": int(args.n_campaigns),
        "seed": int(args.seed),
        "max_window_hours": int(args.max_window_hours),
        "audience_config": args.audience_config,
        "history_valid": {
            "rows": int(len(history_valid)),
            "n_users_indexed": int(len(active_users)),
            "hour_min": int(hv_hour_min),
            "hour_max": int(hv_hour_max),
        },
        "timing_sec": {
            "build_user_index": float(t_index),
            "generate_and_label": float(gen_time),
            "total": float(t_index + gen_time),
        },
        "labels_mean": {
            "at_least_one": float(answers_df["at_least_one"].mean()),
            "at_least_two": float(answers_df["at_least_two"].mean()),
            "at_least_three": float(answers_df["at_least_three"].mean()),
        },
        "labels_p95": {
            "at_least_one": float(answers_df["at_least_one"].quantile(0.95)),
            "at_least_two": float(answers_df["at_least_two"].quantile(0.95)),
            "at_least_three": float(answers_df["at_least_three"].quantile(0.95)),
        },
    }

    with open(out_dir / "report_stage2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_dir / 'offline_campaigns.parquet'}")
    print(f"Wrote: {out_dir / 'offline_answers.parquet'}")
    print(f"Wrote: {out_dir / 'report_stage2.json'}")
    print(f"Total time (sec): {report['timing_sec']['total']:.3f}")


if __name__ == "__main__":
    main()
