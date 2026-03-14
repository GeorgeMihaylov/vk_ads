import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает те же фичи, что и при обучении."""
    features = pd.DataFrame(index=df.index)
    features["cpm"] = df["cpm"].astype(float)
    features["audience_size"] = df["audience_size"].astype(float)
    features["window_length"] = (df["hour_end"] - df["hour_start"] + 1).astype(float)

    def count_pubs(s):
        if pd.isna(s) or s == "":
            return 0
        return len(str(s).split(","))

    features["n_publishers"] = df["publishers"].apply(count_pubs).astype(float)
    features["cpm_per_hour"] = features["cpm"] / features["window_length"]
    return features


def safe_mape(y_true, y_pred, epsilon=1e-8):
    """MAPE с защитой от деления на ноль."""
    # Считаем MAPE только там, где истинное значение достаточно далеко от нуля
    mask = y_true > epsilon
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-dir", type=str, default="artifacts/stage2_full", help="Данные для валидации")
    ap.add_argument("--models-dir", type=str, default="artifacts/stage3", help="Папка с моделями")
    ap.add_argument("--out-dir", type=str, default="artifacts/stage5", help="Папка для отчетов")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stage2_dir = Path(args.stage2_dir)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data to recreate validation set...")
    campaigns = pd.read_parquet(stage2_dir / "offline_campaigns.parquet")
    answers = pd.read_parquet(stage2_dir / "offline_answers.parquet")
    df = campaigns.merge(answers, on="campaign_id")

    print("Extracting features...")
    X = extract_features(df)

    # Повторяем сплит с тем же seed, чтобы точно получить ту же валидационную выборку (X_val)
    _, X_val, _, idx_val = train_test_split(
        X, np.arange(len(X)), test_size=0.2, random_state=args.seed
    )

    df_val = df.iloc[idx_val].copy()

    targets = ["at_least_one", "at_least_two", "at_least_three"]

    detailed_metrics = {}

    print("\nCalculating metrics...\n" + "=" * 50)
    print(f"{'Target':<18} | {'MAE':<8} | {'RMSE':<8} | {'MAPE(%)':<8} | {'R²':<8}")
    print("=" * 50)

    for target in targets:
        y_true = df_val[target].values

        # Загрузка модели
        model = CatBoostRegressor()
        model.load_model(str(models_dir / f"cb_{target}.cbm"))

        # Предсказание
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0.0, 1.0)

        # Расчет метрик
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = safe_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        detailed_metrics[target] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "R2": float(r2)
        }

        print(f"{target:<18} | {mae:.4f}   | {rmse:.4f}   | {mape:.2f}    | {r2:.4f}")

    print("=" * 50)

    # Средние значения по всем таргетам
    mean_mae = np.mean([m["MAE"] for m in detailed_metrics.values()])
    mean_rmse = np.mean([m["RMSE"] for m in detailed_metrics.values()])
    mean_r2 = np.mean([m["R2"] for m in detailed_metrics.values()])

    print(f"{'MEAN (Overall)':<18} | {mean_mae:.4f}   | {mean_rmse:.4f}   | {'-':<8} | {mean_r2:.4f}")

    detailed_metrics["overall"] = {
        "mean_MAE": float(mean_mae),
        "mean_RMSE": float(mean_rmse),
        "mean_R2": float(mean_r2)
    }

    # Сохраняем подробный отчет
    with open(out_dir / "detailed_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed metrics saved to {out_dir / 'detailed_evaluation.json'}")


if __name__ == "__main__":
    main()
