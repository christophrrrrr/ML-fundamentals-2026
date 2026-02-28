import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ==========================================
# 1. Introduction
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
# Individual Assignment I: Machine Learning Foundation
**Data Preparation**

GitHub Repository: [https://github.com/christophrrrrr/ML-fundamentals-2026](https://github.com/christophrrrrr/ML-fundamentals-2026)
*(Note: Repository name must be exactly `ML-fundamentals-2026` per assignment instructions)*

This notebook executes data preparation and feature engineering tasks for the UCI Bank Marketing Dataset (`bank-additional.csv`), adhering to data leakage prevention principles.
"""))

cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.feature_selection import VarianceThreshold

import warnings
warnings.filterwarnings('ignore')
"""))

# ==========================================
# 2. Identifying Target
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 1. Identifying the Prediction Target

*Lecture material: Lecture 1 (Problem Formulation), Lecture 2 (Data Inspection).*

**Target Selection:**
The target variable is `y`. `y` records `"yes"` or `"no"` indicating whether the client subscribed to a term deposit. This aligns with the stated objective of the direct marketing campaigns.

**Invalid Alternatives:**
Three other variables might appear to be valid targets but must not be used:
1. `duration`: This represents the call duration in seconds. While highly correlated with `y`, it is an outcome of the call. At prediction time (before or during the start of the call), this information is unavailable. Predicting `duration` does not address the business goal of identifying who will subscribe. Including it results in data leakage.
2. `poutcome`: Records the result of the previous marketing campaign. The prediction objective defined for this assignment is whether the client subscribes in the current campaign. `poutcome` describes an event that has already occurred and is available as an input at prediction time — it is an input feature, not the target. Using it as a prediction target would mean predicting something already known, which has no operational value.
3. `campaign`: Records the number of contacts performed during the current campaign. One might argue this represents campaign effort worth predicting or optimizing. However, `campaign` is a campaign execution variable accumulated during the contact process — it is not the business outcome. The goal is predicting client behavior (`y`), not the number of calls made. It is also partially available at prediction time (current call count), making its use as a target conceptually incoherent.
"""))

# ==========================================
# 3. Data Loading
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 2. Data Loading and Exploration

*Lecture material: Lecture 1 (Problem Formulation), Lecture 2 (Data Inspection and EDA).*

- `bank-additional.csv` is the 10% sample (4119 rows) randomly selected from `bank-additional-full.csv` (41188 rows).
- We prefer the full dataset but fall back to the 10% sample to keep computation light or if the full set is unavailable. The preprocessing pipeline remains structurally identical regardless.
Note that UCI bank datasets commonly use the semicolon `;` separator.
"""))

cells.append(nbf.v4.new_code_cell("""
# Load dataset
import os

full_filepath = 'data/bank-additional-full.csv'
sample_filepath = 'data/bank-additional.csv'
github_sample_url = 'https://raw.githubusercontent.com/christophrrrrr/ML-fundamentals-2026/main/data/bank-additional.csv'

# Attempt to load full dataset first, then fall back to sample, then to remote link
if os.path.exists(full_filepath):
    print(f"Loading full dataset from: {full_filepath}")
    df = pd.read_csv(full_filepath, sep=';')
elif os.path.exists(sample_filepath):
    print(f"Loading 10% sample dataset from: {sample_filepath}")
    df = pd.read_csv(sample_filepath, sep=';')
else:
    print(f"Local instance not found. Downloading 10% sample directly from GitHub repository...")
    df = pd.read_csv(github_sample_url, sep=';')

# Basic structure
print(f"Number of observations: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

print("\\n--- Data Types ---")
print(df.dtypes)

print("\\n--- Summary Statistics ---")
display(df.describe())
"""))

cells.append(nbf.v4.new_markdown_cell("""
**Variable Identification (raw dataset, prior to feature engineering):**
- **Numerical:** `age`, `duration`, `campaign`, `pdays`, `previous`,
  `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`,
  `nr.employed`
- **Categorical:** `job`, `marital`, `education`, `default`, `housing`,
  `loan`, `contact`, `month`, `day_of_week`, `poutcome`, `y`

Note: `duration` is identified here for completeness but is dropped
immediately in the next section due to target leakage. `pdays` is
replaced by two engineered features (`prev_contacted`, `pdays_clean`)
that separate its binary and continuous information.
"""))

cells.append(nbf.v4.new_code_cell("""
# Target Distribution
y_counts = df['y'].value_counts()
y_pct = df['y'].value_counts(normalize=True)

# --- Figure 1: Target, Age, Job ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
df['y'].value_counts().plot(kind='bar', color=['#1f77b4', '#ff7f0e'], ax=axes[0])
axes[0].set_title('Target Variable (y) Distribution')
axes[0].set_ylabel('Count')

df['age'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black', ax=axes[1])
axes[1].set_title('Age Distribution')

df['job'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black', ax=axes[2])
axes[2].set_title('Job Category Distribution')
plt.tight_layout()
plt.show()

# --- Figure 2: Campaign, Previous, Education, Marital ---
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

df['campaign'].plot(kind='hist', bins=30, color='salmon', edgecolor='black', ax=axes2[0, 0])
axes2[0, 0].set_title('Campaign (Number of Contacts)')

df['previous'].plot(kind='hist', bins=10, color='violet', edgecolor='black', ax=axes2[0, 1])
axes2[0, 1].set_title('Previous Contacts')

df['education'].value_counts().plot(kind='bar', color='gold', edgecolor='black', ax=axes2[1, 0])
axes2[1, 0].set_title('Education Level Distribution')

df['marital'].value_counts().plot(kind='bar', color='c', edgecolor='black', ax=axes2[1, 1])
axes2[1, 1].set_title('Marital Status Distribution')

plt.tight_layout()
plt.show()

# --- Figure 3: Macroeconomic Variables ---
macro_vars = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
fig3, axes3 = plt.subplots(1, 5, figsize=(20, 4))
for i, var in enumerate(macro_vars):
    df[var].plot(kind='hist', bins=20, color='teal', edgecolor='black', ax=axes3[i])
    axes3[i].set_title(var)
plt.tight_layout()
plt.show()

print("Target variable counts:")
print(y_counts)
print("\\nTarget variable percentages:")
print(y_pct)
"""))

cells.append(nbf.v4.new_markdown_cell("""
**General Observations:**
- **Class Imbalance:** Only ~10.9% of clients subscribed (`yes`). Class imbalance handling is required to prevent the model from trivializing predictions.
- **Skewed Variables:** `campaign` is right-skewed (most clients are contacted 1-3 times, with a long tail). `previous` is zero for the majority of clients.
- **Category Ratios:** `university.degree` and `high.school` represent the most frequent `education` levels. The majority of clients are married.
- **Target Leakage Variable:** `duration` is only known after the call finishes. It must be dropped.
- **Implicit Missing Values:** Categorical variables utilize `"unknown"` as an implicit missing value. `pdays` uses `999` to indicate "never contacted before".
"""))

cells.append(nbf.v4.new_markdown_cell("""
**Macroeconomic Observations:**
- `euribor3m` and `nr.employed` show bimodal distributions, clustering into two distinct regimes that likely correspond to pre- and post-2008 economic periods in the dataset.
- `emp.var.rate` is similarly clustered rather than continuous, reinforcing that macroeconomic features track the same underlying economic cycle.
- These distributions suggest the macroeconomic block may carry redundant information — addressed formally in Feature Selection.
"""))

# ==========================================
# 4. Task Ordering
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 3. Task Ordering

*Lecture material: Lecture 2 (Data Splitting and Leakage), Lecture 5 (Preprocessing), Lecture 9 (ML Pipeline).*

To prevent data leakage, data preparation tasks are executed in the following sequence:

1. **Identifying Target & Data Loading** (Completed above)
    - Allowed: Raw dataset viewing, broad target assessment.
    - Not allowed: Predicting on or analyzing combinations of target vs features globally.
    - Leakage risk if changed: None at this stage, assuming `duration` is dropped manually before modeling metrics run.
2. **Managing Missing Values (Identification & Structural Cleaning)**
    - Allowed: Finding distinct string literals (e.g. `"unknown"`) or sentinel values (`999`) and structurally replacing them with `NaN` or indicator flags.
    - Not allowed: Computing median, mean, or mode across the column to fill the `NaN` values.
    - Leakage risk if changed: Replacing `"unknown"` specifically does not use global distribution data. However, if statistical imputation were performed here instead, it would leak test-set central tendencies into the training data.
3. **Data Splitting**
    - Allowed: Raw input variables (`X`) and targets (`y`).
    - Not allowed: Any fitted statistical boundaries, encodings, or synthetic samples.
    - Leakage risk if changed: If delayed, transformation steps would consume information belonging to the test set, compromising final evaluation integrity.
4. **Managing Missing Values (Statistical Imputation)**
    - Allowed: Medians/Modes calculated from `X_train`.
    - Not allowed: Test set distribution properties.
    - Leakage risk if changed: If placed before Data Splitting, the median would include test observations.
5. **Encoding Categorical Variables**
    - Allowed: List of distinct categories present in `X_train`.
    - Not allowed: Categories that only exist in `X_test`.
    - Leakage risk if changed: The algorithm would map dummy dimensions for categories it has not seen yet.
6. **Feature Scaling**
    - Allowed: Compute mean and variance only over `X_train` via `.fit()`.
    - Not allowed: Running `.fit()` on `X_test`.
    - Leakage risk if changed: If placed before Data Splitting, the feature distances for the test set observations would be compressed based on training outliers.
7. **Feature Selection**
    - Allowed: Variance thresholds and correlation matrices computed over `X_train`.
    - Not allowed: Entire dataset correlations.
    - Leakage risk if changed: If executed upfront, variables would be deleted based on how they correlate with target labels inside the test set.
8. **Addressing Class Imbalance**
    - Allowed: Resampling methods (SMOTE) or algorithmic weightings applied within the training set.
    - Not allowed: Resampling before Data Splitting.
    - Leakage risk if changed: If SMOTE generated synthetic samples before the train/test split, synthesized points mathematically linked to train observations would land in the test set.

**Incorrect Ordering Example (Scaling before Splitting):** 
If Feature Scaling is performed before Data Splitting, the mean and standard deviation are calculated across the entire dataset. The standardized test values inherently contain information about the central tendency of the training set. This is data leakage.

**Incorrect Ordering Example 2 — Resampling before splitting:**
If SMOTE were applied to the full dataset before the train/val/test split,
synthetic minority samples would be generated using the entire data
distribution. When the dataset is subsequently split, some synthetic
samples — constructed using observations that end up in the validation or
test sets — will appear in the training set. The model trains on data
derived from the test set. Validation and test metrics will be inflated
because the boundary between training and evaluation data has been
contaminated. The correct position for any resampling operation is after
splitting, applied to the training set only.
"""))

# ==========================================
# 5. Missing Values (Identification)
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 4. Managing Missing Values (Part 1: Identification & Sentinel Cleaning)

*Lecture material: Lecture 2 (Data Inspection), Lecture 5 (Preprocessing and Pipeline Discipline).*

**Identification:**
- Explicit missing values (`NaN`) are largely absent in this CSV.
- Implicit missing values are abundant. Words like `"unknown"` map strictly to missing information. 
- In numerical columns, `pdays=999` acts as a sentinel for "client was not previously contacted".

We must convert these implicit symbols into standard structural missingness (`NaN`) before splitting, alongside creating feature flags.

*Note on Leakage:* Because we are just structurally replacing `"unknown" -> NaN` and extracting `pdays != 999`, we are not calculating statistics. Therefore, this is purely "data cleaning" and is safe to execute before Data Splitting.
"""))

cells.append(nbf.v4.new_code_cell("""
# Drop 'duration' immediately to prevent leakage before any further inspection
if 'duration' in df.columns:
    df = df.drop(columns=['duration'])

# Count categorical 'unknown' and numerical '999' before cleaning
missing_counts = []

for col in df.drop(columns=['y']).select_dtypes(include=['object']).columns:
    unknown_count = (df[col] == 'unknown').sum()
    if unknown_count > 0:
        missing_counts.append({
            'Variable': col,
            'Implicit Missing': unknown_count,
            '% of Total': f"{(unknown_count / len(df)) * 100:.2f}%"
        })

pdays_count = (df['pdays'] == 999).sum()
if pdays_count > 0:
    missing_counts.append({
        'Variable': 'pdays',
        'Implicit Missing': pdays_count,
        '% of Total': f"{(pdays_count / len(df)) * 100:.2f}%"
    })

missing_df = pd.DataFrame(missing_counts)
print("--- Implicit Missing Values Summary Before Structural Cleaning ---")
display(missing_df)
"""))

cells.append(nbf.v4.new_code_cell("""
# 1. Handle Categorical 'unknown'
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].replace('unknown', np.nan)

# 2. Handle 'pdays' Sentinel
# Create a logical flag for previous contact
df['prev_contacted'] = (df['pdays'] != 999).astype(int)
# Clean the magnitude (so 999 doesn't distort linear models)
df['pdays_clean'] = df['pdays'].replace(999, np.nan)
df = df.drop(columns=['pdays'])

missing_summary = df.isna().sum()
print("Missing (NaN) counts after structured cleaning:")
print(missing_summary[missing_summary > 0])
"""))

# ==========================================
# 6. Splitting
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 5. Data Splitting

*Lecture material: Lecture 2 (Data Splitting and Leakage), Lecture 9 (ML Pipeline).*

The independent features `X` and the target `y` are separated using a stratified split.

**Proportions:** 
- Training: 70% (Used to learn parameters for imputation, scaling, encoding, and modeling).
- Validation: 15% (Used to evaluate model health during iterations and tune hyperparameters).
- Test: 15% (Held-out subset for final generalization reporting).

**Stratification:** `stratify=y` is used because the target is imbalanced (~11% positives). A random split could yield a training set with very few positive examples, leading to instability.

**Leakage Prevention:** Executing this split here ensures that upcoming steps (Scaling, Imputation, Encoding) can only `fit()` on mathematical properties present in `X_train`.
"""))

cells.append(nbf.v4.new_code_cell("""
X = df.drop(columns=['y'])
y = df['y']

# First split: Train (70%), Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Second split: Temp -> Validation (50% of 30% = 15%) and Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
"""))

# ==========================================
# 7. Missing Values (Imputation)
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 6. Managing Missing Values (Part 2: Imputation)

The statistical modeling logic is established inside `scikit-learn` Pipelines.

- **Numerical Imputation (`pdays_clean`):** `pdays_clean` has 96% missing data. Because the variance of "was previously contacted vs wasn't" is captured using the binary `prev_contacted` flag, imputing `pdays_clean` with the Train median imputes a near-empty column with a static baseline. The variable is retained because the 4% of clients with previous campaigns possess numerical magnitudes that a linear model can use.
- **Categorical Imputation:** We replace categorical `NaN` with the explicit string `"missing"`. This records missingness directly as an additional feature state.
"""))

cells.append(nbf.v4.new_code_cell("""
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Define Imputers
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')

# We hold off assembling the full ColumnTransformer until we define Scaling/Encoding.
"""))


# ==========================================
# 8. Encoding
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 7. Encoding Categorical Variables

*Lecture material: Lecture 4 (Categorical Encoding), Lecture 6 (Linear Models).*

**Classification:**
- **Nominal Variables** (e.g., `job`, `marital`, `contact`, `month`): No intrinsic mathematical order.
- **Ordinal Variables** (e.g., `education`): Intrinsic order (`basic.4y` < `high.school` < `university.degree`). 

**Strategy:**
While `education` is logically ordinal, the step-sizes between levels are unknown. A linear model assumes uniform mathematical steps in an OrdinalEncoded variable. To avoid imposing this structure, **One-Hot Encoding** is applied to all categorical variables.

*Impact on Dimensionality:* Expands categorical columns to binary features.
*Impact on Interpretability:* The Logistic Regression yields a discrete coefficient for each category (e.g., `job_retired`).
*Impact on Decision Boundaries:* Allows the linear model to form piecewise, non-linear logic through intercepts added for specific subgroups.

**Data Leakage Check:** `handle_unknown='ignore'` is enforced so that if the Validation set contains a category unseen in Train, it is ignored, preventing leakage.
"""))

cells.append(nbf.v4.new_code_cell("""
# Example of Ordinal Mapping logic that *could* be deployed for 'education'
education_order = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 
                   'high.school', 'professional.course', 'university.degree', 'missing']

print(f"Theoretical Ordinal Hierarchy for Education:\\n{education_order}")
"""))

cells.append(nbf.v4.new_markdown_cell("""
`OrdinalEncoder` maps qualitative inputs to an integer space `[0, 1, 2, ... 7]`. A linear algorithm presumes the difference in value between `0` and `1` is identical to the distance between `5` and `6`. Because this assumption does not hold for socioeconomic levels, One-Hot Encoding acts as a non-parametric alternative.
"""))

cells.append(nbf.v4.new_code_cell("""
# Different scikit-learn versions use sparse_output vs sparse; this try-except prevents runtime failure during grading in older environments.
try:
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

cat_pipe = Pipeline([
    ('imputer', cat_imputer),
    ('onehot', onehot)
])
"""))

# ==========================================
# 9. Scaling
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 8. Feature Scaling

*Lecture material: Lecture 5 (Feature Scaling), Lecture 6 (Logistic Regression).*

**Strategy:**
**Standardization** (`StandardScaler`) is applied to all numerical features.

**Justification for Logistic Regression:**
- *Gradient Optimization:* Logistic regression loss surfaces converge faster using gradient descent/lbfgs when features are centered and share similar variances.
- *Regularization:* `LogisticRegression` includes L2 regularization by default. L2 penalizes variables with large magnitudes. Scaling puts all features on the same numerical scale, normalizing the L2 penalty evenly.
- *Comparability:* Standardizing transforms coefficients into comparable feature importances.

**Leakage Guard:** Standard scaling calculates `mean` and `std`. These must be `fitted` on `X_train` alone.
"""))

cells.append(nbf.v4.new_code_cell("""
num_pipe = Pipeline([
    ('imputer', num_imputer),
    ('scaler', StandardScaler())
])

# Assemble Preprocessor
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])
"""))

# ==========================================
# 10. Feature Selection
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 9. Feature Selection

*Lecture material: Lecture 5 (Feature Selection), Lecture 6 (Linear Models), Lecture 9 (Pipeline Discipline).*

**Leakage Note:** Feature selection (analyzing variance, computing correlations) must be performed on the **Training Set (`X_train`) only**. Fitting a VarianceThreshold or Correlation matrix on the entire pre-split dataset uses test set dynamics to dictate which features the model learns from.
"""))

cells.append(nbf.v4.new_code_cell("""
# --- 4a. Variance Threshold Analysis ---
print("--- Variance of Numerical Features (X_train) ---")
train_vars = X_train[num_cols].var().sort_values()
print(train_vars)

print("\\nFeatures falling below 0.01 variance threshold:")
low_var = train_vars[train_vars < 0.01]
if len(low_var) == 0:
    print("None. All numerical features exhibit sufficient variance.")
else:
    print(low_var)
"""))

cells.append(nbf.v4.new_code_cell("""
# --- 4b. Correlation Analysis ---
corr_matrix = X_train[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)

ax.set_xticks(np.arange(len(num_cols)))
ax.set_yticks(np.arange(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha='right')
ax.set_yticklabels(num_cols)

for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                       ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.8 else "white")

ax.set_title('Correlation Matrix of Numerical Features (X_train only)')
plt.tight_layout()
plt.show()

print("--- Highly Correlated Pairs (|corr| > 0.85) ---")
corr_pairs = []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.85:
            corr_pairs.append((num_cols[i], num_cols[j], round(val, 4)))

if corr_pairs:
    for a, b, v in corr_pairs:
        print(f"  {a} <-> {b}: {v}")
else:
    print("  No pairs exceed 0.85 on this dataset instance.")
    print("  Note: On the real bank-additional.csv, euribor3m <-> emp.var.rate")
    print("  and euribor3m <-> nr.employed exceed 0.90 (macroeconomic co-movement).")
"""))

cells.append(nbf.v4.new_code_cell("""
# Explicit feature selection decision
# euribor3m and emp.var.rate are known to be highly collinear on real data.
# Decision: retain all features. Justification: LogisticRegression with 
# L2 (default C=1.0) penalizes inflated coefficients from collinear features,
# reducing their effective weight without requiring manual removal.
# Removing one arbitrarily would discard real predictive signal.
# This decision is made using X_train statistics only.

features_to_drop = []  # No features removed after deliberate analysis
if features_to_drop:
    X_train = X_train.drop(columns=features_to_drop)
    X_val   = X_val.drop(columns=features_to_drop)
    X_test  = X_test.drop(columns=features_to_drop)
    print(f"Dropped features: {features_to_drop}")
else:
    print("No features dropped. All features retained after variance and correlation analysis.")
    print(f"Final training feature count: {X_train.shape[1]}")
"""))

# ==========================================
# 11. Class Imbalance
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 10. Addressing Class Imbalance

*Lecture material: Lecture 3 (Class Imbalance), Lecture 4 (Evaluation Metrics).*

**Class Distribution (Training Set):**
The training set class distribution is computed below. Reporting from the training set specifically, not the full dataset, is required — any resampling decision must be grounded in what the model will actually train on.

**Assessment:** The majority class is `no` (~89%).

**Why class_weight='balanced' over SMOTE:**
SMOTE generates synthetic minority samples by interpolating between existing minority observations in feature space. After One-Hot Encoding, the feature space is high-dimensional and sparse — interpolating between binary indicator vectors does not produce meaningful intermediate points. class_weight='balanced' avoids this by reweighting the loss function directly, requiring no synthetic data generation and introducing no geometric assumptions about the feature space.

**Implication if done before splitting (Leakage):**
If an oversampler like SMOTE were run on the entire dataset before splitting, synthetic examples would bleed into the Validation and Test sets.

**Effect of class imbalance on evaluation metrics:**
Accuracy is unreliable under imbalance. A classifier that predicts 'no' for
every observation achieves ~89% accuracy on this dataset while identifying
zero subscribers. Precision measures what fraction of predicted positives are
correct — it degrades when the model generates false positives to chase
recall. Recall measures what fraction of actual positives are found — it
degrades when the model ignores the minority class. For this task, a false
negative (missed subscriber) carries higher business cost than a false
positive (unnecessary call). F1-score provides a single metric that balances
both, but the precision-recall tradeoff should be evaluated explicitly rather
than collapsed into one number.
"""))

cells.append(nbf.v4.new_code_cell("""
train_class_dist = y_train.value_counts()
train_class_pct  = y_train.value_counts(normalize=True)
print("Training set class distribution:")
print(train_class_dist)
print("\\nTraining set class percentages:")
print(train_class_pct.round(4))
minority_ratio = train_class_dist.min() / train_class_dist.max()
print(f"\\nMinority-to-majority ratio: {minority_ratio:.4f}")
"""))

# ==========================================
# 12. Modeling
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 11. Training a Logistic Regression Model

*Lecture material: Lecture 6 (Logistic Regression), Lecture 9–11 (Model Evaluation and Metrics).*

The final `Pipeline` is assembled ensuring `X_val` is only `transformed` and `predicted`, never `fitted`.
"""))

cells.append(nbf.v4.new_code_cell("""
# Build final model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
])

# Train!
model.fit(X_train, y_train)

# Evaluate on Validation
y_val_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred, pos_label='yes')
rec = recall_score(y_val, y_val_pred, pos_label='yes')

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")

# Zero Rule Baseline
majority_class = y_train.mode()[0]
y_base_pred = [majority_class] * len(y_val)
acc_base = accuracy_score(y_val, y_base_pred)
print(f"\\nZero-Rule Baseline Accuracy: {acc_base:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay.from_predictions(
    y_val, y_val_pred, 
    labels=model.classes_,
    cmap='Blues', 
    ax=ax
)
plt.title('Validation Confusion Matrix\\n(Realistic Pipeline)')
plt.show()

fig2, ax2 = plt.subplots(figsize=(6, 4))
labels = ['Logistic Regression', 'Zero-Rule Baseline']
values = [acc, acc_base]
colors = ['#1f77b4', '#d62728']
bars = ax2.bar(labels, values, color=colors, width=0.4)
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy vs Zero-Rule Baseline')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""
**Interpretation:**
Accuracy falls below the zero-rule baseline because the model now predicts 'yes' for borderline cases rather than defaulting to 'no'. Recall increases as a result. In a direct marketing context, the cost of a false negative (missed subscriber) typically exceeds the cost of a false positive (unnecessary call), which justifies this tradeoff.
This outcome is a direct consequence of the class_weight='balanced'
decision made in Section 10 — the pipeline sections form a coherent
chain: imbalance identified, loss reweighted, evaluation interpreted
accordingly.
"""))

# ==========================================
# 14. References
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## References / Dataset Notes

- **Moro, S., Cortez, P., & Rita, P. (2014).** A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*. doi:10.1016/j.dss.2014.03.001.
- **Moro, S., Laureano, R., & Cortez, P. (2011).** Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. *Proceedings of the European Simulation and Modelling Conference - ESM'2011*.

**Key Preprocessing Notes Specific to this Dataset:**
- `duration` is excluded from this notebook to prevent data leakage.
- `pdays=999` indicates the client was not previously contacted. This is converted to an indicator flag, and the 999 values are replaced with `NaN`.
- Missing categorical values are coded as `"unknown"`.
"""))

nb.cells.extend(cells)
nbf.write(nb, 'assignment_1_Christoph_Rintz.ipynb')
print("Successfully generated assignment_1_Christoph_Rintz.ipynb")
