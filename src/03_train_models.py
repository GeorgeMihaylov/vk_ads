import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает числовые признаки из DataFrame кампаний."""
    features = pd.DataFrame(index=df.index)
    features["cpm"] = df["cpm"].astype(float)
    features["audience_size"] = df["audience_size"].astype(float)
    features["window_length"] = (df["hour_end"] - df["hour_start"] + 1).astype(float)

    # Считаем количество площадок
    # publishers хранятся как строка "1,2,3" или пустая
    def count_pubs(s):
        if pd.isna(s) or s == "":
            return 0
        return len(str(s).split(","))

    features["n_publishers"] = df["publishers"].apply(count_pubs).astype(float)

    # Дополнительные нелинейные фичи, помогающие деревьям
    features["cpm_per_hour"] = features["cpm"] / features["window_length"]

    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-dir", type=str, default="artifacts/stage2_full")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    stage2_dir = Path(args.stage2_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    campaigns = pd.read_parquet(stage2_dir / "offline_campaigns.parquet")
    answers = pd.read_parquet(stage2_dir / "offline_answers.parquet")

    # Слияние по campaign_id
    df = campaigns.merge(answers, on="campaign_id")

    print("Extracting features...")
    X = extract_features(df)
    y_one = df["at_least_one"].values
    y_two = df["at_least_two"].values
    y_three = df["at_least_three"].values

    # Разбиение на Train и Validation
    X_train, X_val, idx_train, idx_val = train_test_split(
        X, np.arange(len(X)), test_size=0.2, random_state=args.seed
    )

    targets = [
        ("at_least_one", y_one),
        ("at_least_two", y_two),
        ("at_least_three", y_three)
    ]

    metrics_report = {}
    models = {}

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    for target_name, y_full in targets:
        print(f"\nTraining model for {target_name}...")
        y_train = y_full[idx_train]
        y_val = y_full[idx_val]

        model = CatBoostRegressor(
            iterations=args.iters,
            learning_rate=0.05,
            depth=6,
            eval_metric='MAE',
            random_seed=args.seed,
            verbose=100
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            use_best_model=True
        )

        preds_val = model.predict(X_val)
        # Ограничиваем предсказания физичным диапазоном [0, 1]
        preds_val = np.clip(preds_val, 0.0, 1.0)

        mae = mean_absolute_error(y_val, preds_val)
        metrics_report[target_name] = {"MAE": float(mae)}

        model_path = out_dir / f"cb_{target_name}.cbm"
        model.save_model(str(model_path))
        models[target_name] = model

        print(f"MAE {target_name}: {mae:.5f}")

    # Считаем средний MAE по трем таргетам
    mean_mae = np.mean([metrics_report[t]["MAE"] for t, _ in targets])
    metrics_report["mean_MAE"] = float(mean_mae)

    print(f"\nOverall Mean MAE: {mean_mae:.5f}")

    # Сохраняем отчет о метриках
    report = {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "metrics": metrics_report,
        "feature_importance": {
            target_name: dict(zip(X.columns, models[target_name].get_feature_importance()))
            for target_name, _ in targets
        }
    }

    with open(out_dir / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved models and metrics to {out_dir}")


if __name__ == "__main__":
    main()
