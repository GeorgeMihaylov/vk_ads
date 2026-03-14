import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clip_monotone(p):
    p = np.clip(p, 0.0, 1.0)
    p[:, 1] = np.minimum(p[:, 1], p[:, 0])
    p[:, 2] = np.minimum(p[:, 2], p[:, 1])
    return p


class OfflineDataset(Dataset):
    def __init__(self, camp_feat, user_idx, targets):
        self.camp_feat = camp_feat.astype(np.float32)
        self.user_idx = user_idx.astype(np.int32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return self.camp_feat.shape[0]

    def __getitem__(self, i):
        return self.camp_feat[i], self.user_idx[i], self.targets[i]


class DeepSetsModel(nn.Module):
    def __init__(self, user_dim, camp_dim, phi_dim=64, hidden=256, out_dim=3):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(user_dim, 128),
            nn.ReLU(),
            nn.Linear(128, phi_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(phi_dim + camp_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, user_feat_batch, camp_feat_batch):
        z = self.phi(user_feat_batch)
        z = z.mean(dim=1)
        x = torch.cat([z, camp_feat_batch], dim=1)
        y = self.rho(x)
        return y


def normalize_fit(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_apply(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def build_batches_user_feat(user_idx_batch, user_feat_table):
    B, K = user_idx_batch.shape
    out = np.empty((B, K, user_feat_table.shape[1]), dtype=np.float32)
    for i in range(B):
        out[i] = user_feat_table[user_idx_batch[i]]
    return out


def eval_on_validate(model, device, user_feat_table, camp_feat, user_idx, targets_true, camp_mean, camp_std):
    model.eval()
    Xc = normalize_apply(camp_feat, camp_mean, camp_std)
    yt = targets_true.astype(np.float64)

    preds = []
    with torch.no_grad():
        bs = 256
        for start in range(0, len(Xc), bs):
            end = min(len(Xc), start + bs)
            c = torch.from_numpy(Xc[start:end]).to(device)
            ui = user_idx[start:end]
            uf = build_batches_user_feat(ui, user_feat_table)
            uf = torch.from_numpy(uf).to(device)

            p = model(uf, c).cpu().numpy().astype(np.float64)
            preds.append(p)

    yp = np.vstack(preds)
    yp = clip_monotone(yp)

    metrics = {}
    for j, name in enumerate(["at_least_one", "at_least_two", "at_least_three"]):
        metrics[name] = {
            "MAE": float(mean_absolute_error(yt[:, j], yp[:, j])),
            "RMSE": rmse(yt[:, j], yp[:, j]),
            "R2": float(r2_score(yt[:, j], yp[:, j])),
        }
    metrics["overall"] = {
        "mean_MAE": float(np.mean([metrics[n]["MAE"] for n in ["at_least_one", "at_least_two", "at_least_three"]])),
        "mean_RMSE": float(np.mean([metrics[n]["RMSE"] for n in ["at_least_one", "at_least_two", "at_least_three"]])),
        "mean_R2": float(np.mean([metrics[n]["R2"] for n in ["at_least_one", "at_least_two", "at_least_three"]])),
    }
    return metrics, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage10-dir", type=str, default="artifacts/stage10")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage11")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    set_seed(args.seed)

    stage10 = Path(args.stage10_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_feat = np.load(stage10 / "user_feat.npy").astype(np.float32)
    offline_camp = np.load(stage10 / "offline_campaign_feat.npy").astype(np.float32)
    offline_ui = np.load(stage10 / "offline_user_idx.npy").astype(np.int32)
    offline_y = np.load(stage10 / "offline_targets.npy").astype(np.float32)

    val_camp = np.load(stage10 / "validate_campaign_feat.npy").astype(np.float32)
    val_ui = np.load(stage10 / "validate_user_idx.npy").astype(np.int32)
    val_y = np.load(stage10 / "validate_targets.npy").astype(np.float32)

    N = offline_camp.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = int(0.1 * N)
    idx_val = idx[:n_val]
    idx_tr = idx[n_val:]

    camp_mean, camp_std = normalize_fit(offline_camp[idx_tr])
    offline_camp_n = normalize_apply(offline_camp, camp_mean, camp_std)

    ds_train = OfflineDataset(offline_camp_n[idx_tr], offline_ui[idx_tr], offline_y[idx_tr])
    ds_val = OfflineDataset(offline_camp_n[idx_val], offline_ui[idx_val], offline_y[idx_val])

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = DeepSetsModel(user_dim=user_feat.shape[1], camp_dim=offline_camp.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    best = {"epoch": -1, "val_mae": 1e9, "state": None}
    t0 = time.time()

    for epoch in range(int(args.epochs)):
        model.train()
        for camp_x, ui, y in dl_train:
            camp_x = camp_x.to(device)
            y = y.to(device)

            uf = build_batches_user_feat(ui.numpy(), user_feat)
            uf = torch.from_numpy(uf).to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(uf, camp_x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for camp_x, ui, y in dl_val:
                camp_x = camp_x.to(device)
                uf = build_batches_user_feat(ui.numpy(), user_feat)
                uf = torch.from_numpy(uf).to(device)
                p = model(uf, camp_x).cpu().numpy()
                preds.append(p)
                trues.append(y.numpy())
        pv = np.vstack(preds).astype(np.float64)
        tv = np.vstack(trues).astype(np.float64)
        pv = clip_monotone(pv)
        val_mae = float(np.mean([mean_absolute_error(tv[:, j], pv[:, j]) for j in range(3)]))

        if val_mae < best["val_mae"]:
            best["val_mae"] = val_mae
            best["epoch"] = epoch
            best["state"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch={epoch} val_mean_MAE={val_mae:.6f} best={best['val_mae']:.6f}")

    model.load_state_dict(best["state"])
    train_time = time.time() - t0

    metrics_validate, pred_validate = eval_on_validate(
        model, device, user_feat, val_camp, val_ui, val_y, camp_mean, camp_std
    )

    pred_path = out_dir / "predictions_deepsets.tsv"
    pd = __import__("pandas")
    pd.DataFrame(pred_validate, columns=["at_least_one", "at_least_two", "at_least_three"]).to_csv(
        pred_path, sep="\t", index=False
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "camp_mean": camp_mean,
            "camp_std": camp_std,
            "user_dim": int(user_feat.shape[1]),
            "camp_dim": int(offline_camp.shape[1]),
            "best_epoch": int(best["epoch"]),
            "best_val_mae_offline10pct": float(best["val_mae"]),
        },
        out_dir / "deepsets.pt"
    )

    report = {
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "train_time_sec": float(train_time),
        "best_epoch_offline10pct": int(best["epoch"]),
        "best_val_mae_offline10pct": float(best["val_mae"]),
        "validate_metrics": metrics_validate,
        "pred_path": str(pred_path),
    }

    with open(out_dir / "report_stage11.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics_validate, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_dir / 'report_stage11.json'}")
    print(f"Wrote: {pred_path}")


if __name__ == "__main__":
    main()
