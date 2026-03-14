import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = ["at_least_one", "at_least_two", "at_least_three"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def clip_monotone(p1, p2, p3):
    p1 = np.clip(p1, 0.0, 1.0)
    p2 = np.clip(p2, 0.0, 1.0)
    p3 = np.clip(p3, 0.0, 1.0)
    p2 = np.minimum(p2, p1)
    p3 = np.minimum(p3, p2)
    return p1, p2, p3


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class TabDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp(X_train, y_train, X_val, y_val, seed, epochs=50, batch_size=2048, lr=1e-3):
    set_seed(seed)
    device = torch.device("cpu")

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    Xtr = (X_train - mean) / std
    Xva = (X_val - mean) / std

    train_ds = TabDataset(Xtr, y_train)
    val_ds = TabDataset(Xva, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.02)
    best = {"epoch": -1, "val_mae": 1e9, "state": None}

    for epoch in range(int(epochs)):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy().reshape(-1)
                preds.append(pred)
        p = np.concatenate(preds, axis=0)
        p = np.clip(p, 0.0, 1.0)
        val_mae = float(mean_absolute_error(y_val, p))
        if val_mae < best["val_mae"]:
            best["val_mae"] = val_mae
            best["epoch"] = epoch
            best["state"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best["state"])
    pack = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "best_epoch": int(best["epoch"]),
        "best_val_mae": float(best["val_mae"]),
    }
    return model, pack


def predict_mlp(model, pack, X):
    mean = pack["mean"]
    std = pack["std"]
    Xn = (X - mean) / std
    ds = TabDataset(Xn, None)
    dl = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in dl:
            p = model(xb).cpu().numpy().reshape(-1)
            preds.append(p)
    p = np.concatenate(preds, axis=0)
    return np.clip(p, 0.0, 1.0)


def load_cb_v2_predictions(models_dir: Path, X_df: pd.DataFrame):
    preds = {}
    for t in TARGETS:
        m = CatBoostRegressor()
        m.load_model(str(models_dir / f"cb_v2_{t}.cbm"))
        preds[t] = m.predict(X_df).astype(np.float64)
    p1, p2, p3 = clip_monotone(preds["at_least_one"], preds["at_least_two"], preds["at_least_three"])
    return {"at_least_one": p1, "at_least_two": p2, "at_least_three": p3}


def grid_blend(y_true, pred_a, pred_b):
    best = {"alpha": None, "mean_mae": 1e9, "by_target": None}
    for alpha in np.linspace(0.0, 1.0, 51):
        by = {}
        maes = []
        for t in TARGETS:
            p = alpha * pred_a[t] + (1.0 - alpha) * pred_b[t]
            maes.append(mean_absolute_error(y_true[t], p))
            by[t] = float(maes[-1])
        mm = float(np.mean(maes))
        if mm < best["mean_mae"]:
            best["mean_mae"] = mm
            best["alpha"] = float(alpha)
            best["by_target"] = by
    return best

def pack_to_jsonable(pack: dict):
    out = {}
    for k, v in pack.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            out[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            out[k] = int(v)
        else:
            out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", type=str, default="artifacts/stage1")
    ap.add_argument("--stage2-dir", type=str, default="artifacts/stage2_full")
    ap.add_argument("--cb-v2-dir", type=str, default="artifacts/stage7")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed(args.seed)

    stage1_dir = Path(args.stage1_dir)
    stage2_dir = Path(args.stage2_dir)
    cb_dir = Path(args.cb_v2_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pub_universe = get_publishers_universe(stage1_dir)

    campaigns = pd.read_parquet(stage2_dir / "offline_campaigns.parquet")
    answers = pd.read_parquet(stage2_dir / "offline_answers.parquet")
    df = campaigns.merge(answers, on="campaign_id", how="inner")

    X_df = build_features(df, pub_universe)
    X = X_df.to_numpy(np.float32)
    y = {t: df[t].astype(np.float32).to_numpy() for t in TARGETS}

    idx = np.arange(len(df))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = int(0.2 * len(idx))
    idx_val = idx[:n_val]
    idx_tr = idx[n_val:]

    X_train = X[idx_tr]
    X_val = X[idx_val]

    nn_models = {}
    nn_packs = {}
    nn_val_preds = {}
    for t in TARGETS:
        model, pack = fit_mlp(
            X_train, y[t][idx_tr],
            X_val, y[t][idx_val],
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        nn_models[t] = model
        nn_packs[t] = pack
        pv = predict_mlp(model, pack, X_val).astype(np.float64)
        nn_val_preds[t] = pv

        torch.save(
            {"state_dict": model.state_dict(), "pack": pack, "in_dim": int(X.shape[1])},
            out_dir / f"mlp_{t}.pt"
        )

    cb_val = load_cb_v2_predictions(cb_dir, X_df.iloc[idx_val])

    y_val_true = {t: y[t][idx_val].astype(np.float64) for t in TARGETS}
    nn_val = {t: nn_val_preds[t] for t in TARGETS}

    blend_offline = grid_blend(y_val_true, cb_val, nn_val)

    validate = pd.read_csv(Path("data") / "validate.tsv", sep="\t")
    Xv_df = build_features(validate, pub_universe)
    Xv = Xv_df.to_numpy(np.float32)

    cb_v = load_cb_v2_predictions(cb_dir, Xv_df)

    nn_v = {}
    for t in TARGETS:
        nn_v[t] = predict_mlp(nn_models[t], nn_packs[t], Xv).astype(np.float64)

    alpha = blend_offline["alpha"]
    p1 = alpha * cb_v["at_least_one"] + (1.0 - alpha) * nn_v["at_least_one"]
    p2 = alpha * cb_v["at_least_two"] + (1.0 - alpha) * nn_v["at_least_two"]
    p3 = alpha * cb_v["at_least_three"] + (1.0 - alpha) * nn_v["at_least_three"]
    p1, p2, p3 = clip_monotone(p1, p2, p3)

    out_pred = pd.DataFrame({"at_least_one": p1, "at_least_two": p2, "at_least_three": p3})
    out_pred.to_csv(out_dir / "predictions_blend.tsv", sep="\t", index=False)

    metrics_validate = None
    answers_path = Path("data") / "validate_answers.tsv"
    if answers_path.exists():
        ans = pd.read_csv(answers_path, sep="\t")
        metrics_validate = {}
        for t in TARGETS:
            yt = ans[t].astype(np.float64).to_numpy()
            yp = out_pred[t].astype(np.float64).to_numpy()
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
        "features_n": int(X.shape[1]),
        "pub_universe_size": int(len(pub_universe)),
        "nn_packs": {t: pack_to_jsonable(nn_packs[t]) for t in TARGETS},
        "blend_offline": blend_offline,
        "validate_metrics": metrics_validate,
    }

    with open(out_dir / "report_stage8.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report.get("validate_metrics", {}), ensure_ascii=False, indent=2))
    print(f"Wrote: {out_dir / 'predictions_blend.tsv'}")
    print(f"Wrote: {out_dir / 'report_stage8.json'}")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    main()
