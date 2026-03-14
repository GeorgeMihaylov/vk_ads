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


TARGET_NAMES = ["at_least_one", "at_least_two", "at_least_three"]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clip_monotone_np(p):
    p = np.clip(p, 0.0, 1.0)
    p[:, 1] = np.minimum(p[:, 1], p[:, 0])
    p[:, 2] = np.minimum(p[:, 2], p[:, 1])
    return p


def metrics_np(y_true, y_pred):
    out = {}
    for j, name in enumerate(TARGET_NAMES):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        out[name] = {
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": rmse(yt, yp),
            "R2": float(r2_score(yt, yp)),
        }
    out["overall"] = {
        "mean_MAE": float(np.mean([out[n]["MAE"] for n in TARGET_NAMES])),
        "mean_RMSE": float(np.mean([out[n]["RMSE"] for n in TARGET_NAMES])),
        "mean_R2": float(np.mean([out[n]["R2"] for n in TARGET_NAMES])),
    }
    return out


def norm_fit(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def norm_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)


class NpyDataset(Dataset):
    def __init__(self, camp_feat, user_idx, y):
        self.camp_feat = camp_feat.astype(np.float32)
        self.user_idx = user_idx.astype(np.int64)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.camp_feat.shape[0]

    def __getitem__(self, i):
        return self.camp_feat[i], self.user_idx[i], self.y[i]


class CondAttnPooling(nn.Module):
    def __init__(self, z_dim: int, c_dim: int, n_heads: int = 6, head_dim: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.key = nn.Linear(z_dim, n_heads * head_dim, bias=False)
        self.query = nn.Linear(c_dim, n_heads * head_dim, bias=False)
        self.scale = (head_dim ** 0.5)

    def forward(self, z, c, mask=None):
        B, K, _ = z.shape
        k = self.key(z).view(B, K, self.n_heads, self.head_dim)  # [B,K,H,D]
        q = self.query(c).view(B, self.n_heads, self.head_dim)   # [B,H,D]
        logits = (k * q[:, None, :, :]).sum(dim=-1) / self.scale  # [B,K,H]
        if mask is not None:
            logits = logits.masked_fill(~mask[:, :, None], -1e9)
        w = torch.softmax(logits, dim=1)                          # [B,K,H]
        pooled = (w[:, :, :, None] * k).sum(dim=1)                # [B,H,D]
        return pooled.reshape(B, self.n_heads * self.head_dim)    # [B, H*D]


class HeavyDeepSets(nn.Module):
    def __init__(self, user_dim: int, camp_dim: int, z_dim: int = 192,
                 n_heads: int = 6, head_dim: int = 64, rho_hidden: int = 768):
        super().__init__()
        self.user_norm = nn.LayerNorm(user_dim)
        self.camp_norm = nn.LayerNorm(camp_dim)

        self.phi = nn.Sequential(
            nn.Linear(user_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.03),
            nn.Linear(384, z_dim),
            nn.ReLU(),
        )

        self.attn = CondAttnPooling(z_dim=z_dim, c_dim=camp_dim, n_heads=n_heads, head_dim=head_dim)

        rho_in = (n_heads * head_dim) + z_dim + z_dim + camp_dim  # attn + mean + max + campaign
        self.rho = nn.Sequential(
            nn.Linear(rho_in, rho_hidden),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(rho_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, user_x, camp_x, mask=None):
        camp_x = self.camp_norm(camp_x)
        user_x = self.user_norm(user_x)
        z = self.phi(user_x)  # [B,K,z_dim]

        if mask is None:
            z_mean = z.mean(dim=1)
            z_max = z.max(dim=1).values
        else:
            m = mask[:, :, None].float()
            denom = m.sum(dim=1).clamp(min=1.0)
            z_mean = (z * m).sum(dim=1) / denom
            z_masked = z.masked_fill(~mask[:, :, None], -1e9)
            z_max = z_masked.max(dim=1).values

        z_attn = self.attn(z, camp_x, mask=mask)
        x = torch.cat([z_attn, z_mean, z_max, camp_x], dim=1)
        return self.rho(x)


def make_mask_and_gather(user_idx: torch.Tensor, user_table: torch.Tensor):
    mask = user_idx >= 0
    idx = user_idx.clamp(min=0)
    user_x = user_table[idx]  # [B,K,D]
    user_x = user_x * mask[:, :, None].float()
    return user_x, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage10-dir", type=str, default="artifacts/stage10_K256")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage13_K256")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", type=str, default="auto")  # auto|cpu|cuda
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    set_seed(args.seed)

    stage10 = Path(args.stage10_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_feat = np.load(stage10 / "user_feat.npy").astype(np.float32)
    off_camp = np.load(stage10 / "offline_campaign_feat.npy").astype(np.float32)
    off_ui = np.load(stage10 / "offline_user_idx.npy").astype(np.int64)
    off_y = np.load(stage10 / "offline_targets.npy").astype(np.float32)

    val_camp = np.load(stage10 / "validate_campaign_feat.npy").astype(np.float32)
    val_ui = np.load(stage10 / "validate_user_idx.npy").astype(np.int64)
    val_y = np.load(stage10 / "validate_targets.npy").astype(np.float32)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # offline split 10%
    N = off_camp.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = int(0.1 * N)
    idx_val = idx[:n_val]
    idx_tr = idx[n_val:]

    camp_mean, camp_std = norm_fit(off_camp[idx_tr])
    off_camp_n = norm_apply(off_camp, camp_mean, camp_std)
    val_camp_n = norm_apply(val_camp, camp_mean, camp_std)

    user_mean, user_std = norm_fit(user_feat)
    user_feat_n = norm_apply(user_feat, user_mean, user_std)

    ds_tr = NpyDataset(off_camp_n[idx_tr], off_ui[idx_tr], off_y[idx_tr])
    ds_va = NpyDataset(off_camp_n[idx_val], off_ui[idx_val], off_y[idx_val])
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)

    user_table = torch.from_numpy(user_feat_n).to(device)
    model = HeavyDeepSets(user_dim=user_feat.shape[1], camp_dim=off_camp.shape[1]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = {"epoch": -1, "val_mae": 1e9, "state": None}
    t0 = time.time()

    for epoch in range(int(args.epochs)):
        model.train()
        for camp_x, ui, y in dl_tr:
            camp_x = camp_x.to(device, non_blocking=True)
            ui = ui.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            user_x, mask = make_mask_and_gather(ui, user_table)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(user_x, camp_x, mask=mask)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for camp_x, ui, y in dl_va:
                camp_x = camp_x.to(device, non_blocking=True)
                ui = ui.to(device, non_blocking=True)
                user_x, mask = make_mask_and_gather(ui, user_table)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    p = model(user_x, camp_x, mask=mask).float().cpu().numpy()
                preds.append(p)
                trues.append(y.numpy())

        pv = np.vstack(preds).astype(np.float64)
        tv = np.vstack(trues).astype(np.float64)
        pv = clip_monotone_np(pv)
        val_mae = float(np.mean([mean_absolute_error(tv[:, j], pv[:, j]) for j in range(3)]))

        if val_mae < best["val_mae"]:
            best["val_mae"] = val_mae
            best["epoch"] = epoch
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch={epoch} offline10pct_val_meanMAE={val_mae:.6f} best={best['val_mae']:.6f}")

    model.load_state_dict(best["state"])
    train_time = time.time() - t0

    # validate inference (1008)
    model.eval()
    with torch.no_grad():
        camp_x = torch.from_numpy(val_camp_n).to(device)
        ui = torch.from_numpy(val_ui).to(device)
        user_x, mask = make_mask_and_gather(ui, user_table)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(user_x, camp_x, mask=mask).float().cpu().numpy().astype(np.float64)

    pred = clip_monotone_np(pred)
    m = metrics_np(val_y.astype(np.float64), pred)

    pred_path = out_dir / "predictions_deepsets_attn.tsv"
    pd = __import__("pandas")
    pd.DataFrame(pred, columns=TARGET_NAMES).to_csv(pred_path, sep="\t", index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "camp_mean": camp_mean,
            "camp_std": camp_std,
            "user_mean": user_mean,
            "user_std": user_std,
            "best_epoch_offline10pct": int(best["epoch"]),
            "best_val_mae_offline10pct": float(best["val_mae"]),
            "k": int(off_ui.shape[1]),
            "user_dim": int(user_feat.shape[1]),
            "camp_dim": int(off_camp.shape[1]),
        },
        out_dir / "deepsets_attn.pt"
    )

    report = {
        "stage10_dir": str(stage10),
        "device": str(device),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "K": int(off_ui.shape[1]),
        "train_time_sec": float(train_time),
        "best_epoch_offline10pct": int(best["epoch"]),
        "best_val_mae_offline10pct": float(best["val_mae"]),
        "validate_metrics": m,
        "pred_path": str(pred_path),
    }
    with open(out_dir / "report_stage13.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_dir / 'report_stage13.json'}")
    print(f"Wrote: {pred_path}")


if __name__ == "__main__":
    main()
