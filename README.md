# VK Ads Reach Prediction (контрфактическое моделирование + DeepSets)

## Задача
Для каждой рекламной кампании из `validate.tsv` предсказать три вероятности:
- `at_least_one`  — доля пользователей, увидевших рекламу ≥ 1 раза
- `at_least_two`  — доля пользователей, увидевших рекламу ≥ 2 раза
- `at_least_three`— доля пользователей, увидевших рекламу ≥ 3 раза

Все значения — в диапазоне [0, 1] и должны удовлетворять:
`at_least_one >= at_least_two >= at_least_three`.

## Данные
Используются предоставленные организаторами таблицы:
- `history_train` (показы победителей аукциона)
- `sessions_train` (сессии пользователей)
- `train` / `train_answers`
- `validate` / `validate_answers` (для локальной оценки)

Особенность: в `history_train` есть только победившие ставки, поэтому прямого supervised-обучения “как в лоб” недостаточно.

## Ключевая идея решения
### 1) Контрфактическая офлайн-разметка
Для обучения мы генерируем синтетические “виртуальные кампании” на отложенном периоде истории и симулируем участие в аукционе по правилам:
- ставка выше победившей → выигрываем показ
- ставка равна → выигрываем с вероятностью 50%
- учитываем правило сессии (повторные показы в одной сессии ограничены)

Так получаем офлайн-датасет (120k кампаний) с таргетами `at_least_one/two/three`.

### 2) Фичи кампании (campaign features)
Извлекаем физически осмысленные признаки:
- `cpm`, `log_cpm`, `cpm_per_hour`
- `window_length`, `hod_start/hod_end`, распределение часов по суткам (24 признака)
- one-hot по выбранным площадкам (publishers)
- размер аудитории и др.

### 3) User feature store
Строим вектор признаков на каждого пользователя по `history_train` и `sessions_train`:
- активность по часам суток (24)
- предпочтения по площадкам (21)
- агрегаты по CPM, числу дней активности, сессиям, длине сессий и т.д.
Итоговая размерность user-вектора: 70.

### 4) Итоговая модель: DeepSets + Attention pooling (K=256)
Так как в кампании задано множество `user_ids` (порядок не важен), используем set-модель:
- `phi(user)` → эмбеддинг пользователя
- attention pooling по множеству пользователей (веса зависят от признаков кампании)
- `rho([pooled_user_emb || campaign_features])` → 3 таргета одновременно

Для каждого примера берём случайную подвыборку K=256 пользователей из аудитории.
После инференса применяем постобработку:
- clip в [0,1]
- монотонность `p1>=p2>=p3`

## Метрики (локальная проверка на validate_answers)
Финальная модель (DeepSets-attn, K=256) показала:
- overall mean MAE ≈ 0.02687
- overall mean RMSE ≈ 0.05761
- overall mean R² ≈ 0.77376

(Расчёт по `data/validate_answers.tsv`.)

## Воспроизведение (основные команды)
1) Обработка исходных данных:
```bash
python src/01_prepare_data.py --data-dir data --out-dir artifacts/stage1
Контрфактическая генерация offline-датасета:
```
```bash
python src/02_make_offline_dataset.py --stage1-dir artifacts/stage1 --out-dir artifacts/stage2_full
User feature store:
```

```bash
python src/09_build_user_features.py --stage1-dir artifacts/stage1 --out-dir artifacts/stage9
Подготовка датасета для DeepSets (K=256):
```
```bash
python src/10_prepare_deepsets_datasets.py --stage1-dir artifacts/stage1 --stage2-dir artifacts/stage2_full --stage9-dir artifacts/stage9 --out-dir artifacts/stage10_K256 --K 256 --seed 42
Обучение DeepSets-attn и предсказания:
```
```bash
python src/13_train_deepsets_attention_k256.py --stage10-dir artifacts/stage10_K256 --out-dir artifacts/stage13_K256 --device cpu --epochs 30 --batch-size 128
Финальные предсказания для сдачи:
Файл predictions_final.tsv формируется в artifacts/stage12_attnK256/.
```

## Сравнение подходов (локальная проверка на validate_answers.tsv)

Ниже — сравнение качества разных моделей на одном и том же `validate_answers.tsv` (1008 кампаний), метрики усреднены по трём таргетам (`at_least_one/two/three`). 

| Подход | mean MAE | mean RMSE | mean R² | Комментарий |
|---|---:|---:|---:|---|
| CatBoost v2 (табличные фичи) | 0.030336 | 0.070531 | 0.654880 | Сильный табличный бейзлайн |
| DeepSets (K=64, mean pooling) | 0.034468 | 0.074651 | 0.615426 | Слабее CatBoost, мало “сигнала” из аудитории |
| Blend: 0.95 * CatBoost v2 + 0.05 * DeepSets(K=64) | 0.030315 | 0.070381 | 0.656742 | Микро-улучшение за счёт независимого сигнала |
| DeepSets + conditioned attention pooling (K=256) | **0.026866** | **0.057610** | **0.773757** | Лучший результат, существенный прирост |
| Blend: CatBoost v2 + DeepSets-attn(K=256) | best_alpha = 0.0 | — | — | Лучший бленд = чистый DeepSets-attn(K=256) |

### Метрики по каждому таргету (MAE / RMSE / R²)

| Модель | at_least_one | at_least_two | at_least_three |
|---|---|---|---|
| CatBoost v2 | MAE 0.04331 / RMSE 0.08187 / R² 0.68589 | MAE 0.02709 / RMSE 0.06841 / R² 0.66248 | MAE 0.02061 / RMSE 0.06131 / R² 0.61627 |
| DeepSets (K=64, mean pooling) | MAE 0.04787 / RMSE 0.08755 / R² 0.64081 | MAE 0.03154 / RMSE 0.07306 / R² 0.61501 | MAE 0.02399 / RMSE 0.06334 / R² 0.59046 |
| DeepSets-attn (K=256, финальная) | **MAE 0.03875 / RMSE 0.07071 / R² 0.76569** | **MAE 0.02409 / RMSE 0.05523 / R² 0.77999** | **MAE 0.01776 / RMSE 0.04689 / R² 0.77559** |

Примечание: финальный бленд с CatBoost для DeepSets-attn(K=256) не дал улучшения (best_alpha = 0.0), поэтому финальный сабмит = чистый DeepSets-attn(K=256).


### Итог
Финальная модель для сдачи — **DeepSets-attn(K=256)** (в бленде оптимальный вес CatBoost оказался 0), файл для сабмита: `predictions_final.tsv`.



## Для проверки предоставляется архив с кодом и итоговым `predictions_final.tsv`.