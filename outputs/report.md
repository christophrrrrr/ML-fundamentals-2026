# Bank Marketing: Data Preparation & Feature Engineering Pipeline

## 1. Dataset & Task
- **Domain:** Direct marketing phone-call campaigns for term deposits.
- **Dataset Facts:** Used UCI Bank Marketing dataset. Target `y` indicates subscription ('yes'/'no').
- **Moro et al. Citations:** Moro et al., 2014 (bank-additional); Moro et al., 2011 (bank-full).

## 2. Leakage Policy & Duration
- **Decision:** Excluded `duration` from the main model features.
- **Reason:** Duration is highly predictive but not known before a call is performed. Including it violates realism and constitutes target leakage.
- **Ablation:** A benchmark model *with* duration was trained to demonstrate the artificial performance inflation caused by this leakage.

## 3. Preprocessing Decisions
- **Missingness ("unknown"):** Explicitly converted "unknown" categorical values to explicit missingness (replaced with `NaN` before split, imputed with constant `"missing"` after split). This honors the lack of info without dropping rows.
- **Pdays Sentinel Logic:** Sentinel values (999 or -1) indicating 'never contacted' were extracted into a binary feature `prev_contacted`, and the sentinel was converted to `NaN` in `pdays_clean` to prevent its magnitude distorting linear models.
- **Split Strategy:** Used `stratified_random` to maintain minority class distributions, performed *before* any statistical imputation or scaling to prevent data leakage.
- **Pipeline Tools:** Used `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` within a `scikit-learn` `ColumnTransformer`.

## 4. Modeling & Imbalance Handling
- **Algorithm:** Logistic Regression (used as a simple baseline sanity check, no heavy hyperparameter tuning).
- **Imbalance handling:** Set `class_weight="balanced"` to penalize minority class misclassifications symmetrically.

## 5. Main Results (Realistic Model without Duration)
- **Balanced Accuracy:** 0.7497
- **ROC AUC:** 0.8001
- **Baseline Strategy (Predict Majority):** Balanced Acc: 0.5000

## 6. Duration Benchmark (Unrealistic Leakage Model)
- **Balanced Accuracy (with duration):** 0.8838
- **ROC AUC (with duration):** 0.9402
- **Observation:** Notice the inflated metric due to dataset leakage.
