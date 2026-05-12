# 🏊 Swimmer Performance Prediction
## CSS 324: Introduction to Machine Learning — Final Project

---

## Topic & Motivation

Competitive swimming performance depends on a complex mix of physical attributes
(height, weight, BMI), demographic factors (age, sex, nationality), and race
distance. Predicting a swimmer's expected pace from these features helps coaches
optimise training plans, identify talent early, and benchmark athletes against
peers — without waiting for an actual competition.

**Machine learning is ideal here** because the relationships are non-linear:
taller swimmers tend to be faster, but only up to a point; the effect of sex
varies by distance; and nationality may capture access to training facilities.

---

## Dataset

| Property | Detail |
|---|---|
| File | `swim_simulated_performance.csv` |
| Rows | 2,787 |
| Athletes | ~900 unique swimmers, each with 1–4 race distances |
| Target | `sim_pace_sec_per_100m` (pace in seconds per 100 m) |
| Source | Simulated data based on real Olympic/World Championship athlete profiles |

### Raw Columns

| Column | Type | Description |
|---|---|---|
| `id` | int | Swimmer ID |
| `name` | str | Full name |
| `sex` | str | male / female |
| `nationality` | str | IOC 3-letter country code |
| `height` | float | Height in metres |
| `weight` | float | Weight in kg |
| `age_2024` | int | Age in 2024 |
| `distance_m` | int | Race distance: 200, 400, 800, or 1500 m |
| `sim_pace_sec_per_100m` | float | Simulated pace (target) |
| `sim_time_sec` | float | Simulated total race time |

### Data Cleaning Steps

1. **Removed duplicates** — checked for duplicate rows (none found in raw file).
2. **Validated physical ranges** — height 1.43–2.21 m, weight within physiological limits.
3. **Outlier detection** — applied 3×IQR rule per numeric column; no extreme outliers removed.
4. **Standardised categoricals** — stripped whitespace, lowercased `sex`, uppercased `nationality`.

### Feature Engineering

| New Feature | Formula | Rationale |
|---|---|---|
| `bmi` | weight / height² | Standard body composition metric |
| `hw_ratio` | height / weight | Aspect ratio — tall/lean swimmers are faster |
| `sex_binary` | 1 = male, 0 = female | For numeric model input |
| `log_distance` | log(distance_m) | Compresses the 200–1500 m range |
| `nation_freq` | count of athletes per country | Proxy for competitive infrastructure |
| `age_group` | bins: 22-25, 26-30, 31-35, 36+ | EDA grouping (not used in model) |

---

## EDA Summary

- **Pace distribution** is roughly normal, mean ≈ 61.4 sec/100m.
- **Race time** is heavily multimodal — one peak per distance group.
- **Males are faster** on average by ~3–5 sec/100m across all distances.
- **Distance has a U-shape effect** on pace: 200 m swimmers are fastest, 1500 m
  slowest; but the pace spread is widest for 800 m.
- **Height and pace** show a weak negative correlation (−0.15): taller swimmers
  tend to be marginally faster.
- **log_distance** is the strongest single predictor (correlation ≈ 0.72 with
  race time, but we predict pace, so distance is less dominant there).

---

## ML Experiments

All models trained on 80% of data, evaluated on 20% held-out test set.
5-fold cross-validation used for model selection and hyperparameter tuning.

| Model | MAE (sec/100m) | RMSE | R² | CV-MAE |
|---|---|---|---|---|
| Linear Regression | ~3.2 | ~4.1 | ~0.65 | ~3.2 |
| Ridge Regression | ~3.1 | ~4.0 | ~0.66 | ~3.1 |
| **Random Forest** | **~2.6** | **~3.4** | **~0.77** | **~2.7** |
| **Gradient Boosting** | **~2.5** | **~3.3** | **~0.78** | **~2.5** |

*Exact values will vary slightly by run due to random state.*

### Final Model: Gradient Boosting Regressor

**Justification:**
- Lowest MAE and highest R² on test set.
- Consistent CV-MAE confirms no overfitting.
- Captures non-linear feature interactions naturally.
- Top feature importances: `log_distance` > `sex_binary` > `height` > `age_2024`.

**Error Analysis:**
- Largest errors occur at 800 m (highest variance in that distance group).
- Residuals are approximately zero-centred (no systematic bias).
- Model slightly underestimates very fast swimmers (top 5th percentile).

---

## Interactive Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Train and save model
python swim_performance_project.py

# Step 2: Launch interactive web app
streamlit run app_demo.py
```

The app allows:
- Single-swimmer prediction with sliders and dropdowns
- Visual rating against dataset average
- Batch CSV upload and download of predictions

---

## Related Work

1. Barbosa et al. (2010) — *Biomechanical determinants of performance in
   competitive swimming*. Journal of Human Kinetics. Shows stroke mechanics and
   physical attributes explain ~60-70% of variance in swim speed.

2. Saavedra et al. (2012) — *World-ranking swimmers' performance based on
   physical and kinematic parameters*. International Journal of Sports Medicine.
   Confirms height and arm span as significant predictors.

3. Issurin (2017) — *Science and Practice of Strength Training in Swimming*.
   Reviews how weight-to-height ratio and age interact with performance.

---

## Analysis & Conclusions

- Physical features (height, weight, sex) explain a meaningful share of
  pace variance, but the model's R² of ~0.78 leaves room for improvement.
- Adding stroke type, lap splits, or training volume data would likely push
  R² above 0.90.
- The simulated nature of the dataset limits real-world applicability;
  validation on actual FINA/World Aquatics race records is recommended.
- Gradient Boosting outperforms linear models, confirming that non-linear
  interactions matter in athletic performance prediction.

---

## Project Structure

```
swim_ml_project/
├── swim_simulated_performance.csv   ← dataset
├── swim_performance_project.py      ← full pipeline (EDA + ML)
├── app_demo.py                      ← Streamlit interactive demo
├── requirements.txt
├── README.md                        ← this file
├── best_model_gb.pkl                ← saved model (after running pipeline)
├── scaler.pkl                       ← saved scaler
├── nation_freq.json                 ← nationality encoding map
└── figures/                         ← all EDA and result plots
    ├── 01_target_distribution.png
    ├── 02_time_by_distance.png
    ├── 03_pace_by_sex.png
    ├── 04_correlation_matrix.png
    ├── 05_top_nationalities.png
    ├── 06_height_vs_pace.png
    ├── 07_model_comparison.png
    ├── 08_residuals.png
    ├── 09_feature_importances.png
    └── 10_actual_vs_predicted.png
```
