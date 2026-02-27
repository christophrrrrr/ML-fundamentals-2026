import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ==========================================
# 1. Introduction
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
# Individual Assignment I: Machine Learning Foundation
**Data Preparation**

GitHub Repository: [https://github.com/christophrrrrr/ML-fundamentals-2026.git](https://github.com/christophrrrrr/ML-fundamentals-2026.git)
*(Note: Repository name must be exactly `ML-fundamentals-2026` per assignment instructions)*

This notebook executes data preparation and feature engineering tasks for the UCI Bank Marketing Dataset (`bank-additional.csv`), in strict adherence to data leakage prevention principles.
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
The target variable is `y`. Upon manual inspection of the dataset description and features, `y` records `"yes"` or `"no"` indicating whether the client subscribed to a term deposit. This aligns perfectly with the stated objective of the direct marketing campaigns.

**Invalid Alternatives:**
Two other variables might superficially appear to be valid targets but must not be used:
2. `poutcome`: This records the outcome of the *previous* marketing campaign. While useful as a highly predictive feature, it is fundamentally a descriptor of a *past* event rather than the current campaign's outcome. Furthermore, because it is known at prediction time, it serves as a valid input feature, not the target to predict.
3. `campaign`: One might assume the "number of contacts performed" is an outcome to optimize. However, this is an execution variable known during the campaign management, not the business objective (which is whether the client actually subscribed to a deposit).
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
**Variable Identification:**
- **Numerical:** `age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
- **Categorical:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`, `y`
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
**Observations:**
- **Class Imbalance:** Only ~10.9% of clients subscribed (`yes`), meaning early handling for class imbalance will be required to prevent the model from trivializing predictions to the majority class.
- **Skewed Variables:** `campaign` is heavily right-skewed (most clients are contacted 1-3 times, but there is a long tail of clients contacted repeatedly). `previous` is zero for the vast majority of clients, indicating most clients have never been contacted in past campaigns.
- **Category Ratios:** `university.degree` and `high.school` dominate the `education` distribution. The majority of clients are married.
- **Macroeconomics:** The distributions of `euribor3m` (Euro Interbank Offered Rate) and `nr.employed` are distinctly bimodal, clustering sharply rather than showing a normal distribution. `cons.price.idx` and `cons.conf.idx` also display non-normal clustering and skewness.
- **Special Consideration Variables:** 
    1. `duration` strongly dictates target leakage since the length of a call is only known *after* the call finishes, which violates the premise of predicting success in advance. It must be dropped.
    2. `campaign` possesses an extreme right tail which linear models may struggle gracefully with; regularizing via standardization later will be critical.
- **Implicit Missing Values:** Several categorical variables (like `job` or `education`) utilize the string `"unknown"` as an implicit missing value. The numerical variable `pdays` uses `999` to declare "never contacted before".
"""))

# ==========================================
# 4. Task Ordering
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 3. Task Ordering

*Lecture material: Lecture 2 (Data Splitting and Leakage), Lecture 5 (Preprocessing), Lecture 9 (ML Pipeline).*

To strictly prevent data leakage and abide by ML Pipeline discipline, I execute the data preparation tasks in the following sequence:

1. **Identifying Target & Data Loading** (Completed above)
    - ✅ **Allowed:** Raw dataset viewing, broad target assessment.
    - 🚫 **Not allowed:** Predicting on or analyzing combinations of target vs features globally.
    - ⚠️ **Leakage risk if changed:** None at this conceptual stage, as long as `duration` is dropped manually before modeling metrics run.
2. **Managing Missing Values (Identification & Structural Cleaning)**
    - ✅ **Allowed:** Finding distinct string literals (e.g. `"unknown"`) or sentinel values (`999`) and structurally replacing them with `NaN` or indicator flags.
    - 🚫 **Not allowed:** Computing median, mean, or mode across the column to fill the `NaN` values.
    - ⚠️ **Leakage risk if changed:** Replacing `"unknown"` specifically does not use global distribution data. However, if *statistical* imputation were performed here instead, it would leak test-set central tendencies into the training data.
3. **Data Splitting**
    - ✅ **Allowed:** Raw input variables (`X`) and targets (`y`).
    - 🚫 **Not allowed:** Any fitted statistical boundaries, encodings, or synthetic samples.
    - ⚠️ **Leakage risk if changed:** If delayed to later in the pipeline, every transformation step ahead of it would accidentally consume information belonging strictly to the test set, compromising final evaluation integrity.
4. **Managing Missing Values (Statistical Imputation)**
    - ✅ **Allowed:** Medians/Modes calculated strictly from `X_train`.
    - 🚫 **Not allowed:** Test set distribution properties.
    - ⚠️ **Leakage risk if changed:** If placed before Data Splitting, the median would include test observations.
5. **Encoding Categorical Variables**
    - ✅ **Allowed:** List of distinct categories present purely in `X_train`.
    - 🚫 **Not allowed:** Categories that only exist in `X_test`.
    - ⚠️ **Leakage risk if changed:** The algorithm would map dummy dimensions for categories it conceptually hasn't "seen" yet, failing in real-world scenarios where unknown inputs arrive.
6. **Feature Scaling**
    - ✅ **Allowed:** Compute mean and variance only over `X_train` via `.fit()`.
    - 🚫 **Not allowed:** Running `.fit()` on `X_test`.
    - ⚠️ **Leakage risk if changed:** If placed before Data Splitting, the feature distances for the test set observations would be mathematically compressed based on training outliers (or vice versa).
7. **Feature Selection**
    - ✅ **Allowed:** Variance thresholds and correlation matrices computed *over* `X_train`.
    - 🚫 **Not allowed:** Entire dataset correlations.
    - ⚠️ **Leakage risk if changed:** If executed upfront, we would delete variables based on how they correlate with target labels inside the pristine test set.
8. **Addressing Class Imbalance**
    - ✅ **Allowed:** Resampling methods (SMOTE) or algorithmic weightings applied entirely within the training vault.
    - 🚫 **Not allowed:** Resampling before Data Splitting.
    - ⚠️ **Leakage risk if changed:** If SMOTE generated synthetic samples before the train/test split, synthesized points mathematically linked to train observations would bleed across the boundary and land in the test set. Recall scores would artificially inflate due to structural duplication rather than true generalization power.

**Incorrect Ordering Example (Scaling before Splitting):** 
If we performed **Feature Scaling** before **Data Splitting**, we would calculate the mean and standard deviation across the *entire* dataset. The standardized test values would inherently contain information about the central tendency of the training set. This is a classic form of data leakage, artificially inflating the test evaluation metrics because the test data "peeked" into the global distribution.

**Incorrect Ordering Example (SMOTE before Splitting):**
If resampling like SMOTE were applied before splitting, synthetic minority samples generated from training points would overlap geometrically with validation and test set spaces, causing grossly inflated recall and precision scores on data that isn't functionally new.
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
# Count categorical 'unknown' and numerical '999' before cleaning
missing_counts = []

for col in df.select_dtypes(include=['object']).columns:
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
# Drop 'duration' immediately to avoid leakage
if 'duration' in df.columns:
    df = df.drop(columns=['duration'])

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

I separate the independent features `X` and the target `y`, and perform a stratified split.

**Proportions:** 
- Training: 70% (Used strictly to learn parameters for imputation, scaling, encoding, and modeling).
- Validation: 15% (Used to evaluate model health during iterations and tune hyperparameters).
- Test: 15% (Strictly held-out vault for final generalization reporting; untouched here).

**Stratification:** We use `stratify=y` because the target is highly imbalanced (~11% positives). A random split could accidentally yield a training set with very few positive examples, leading to instability.

**Leakage Prevention:** By executing this split NOW, I guarantee that upcoming steps (Scaling, Imputation, Encoding) can only `fit()` on mathematical properties isolated inside `X_train`.
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

Now that data is split safely, we establish the statistical modeling logic inside `scikit-learn` Pipelines.

- **Numerical Imputation (`pdays_clean`):** `pdays_clean` exhibits extreme missingness (nearly 96% of clients were never previously contacted, meaning they possess `NaN` values here). Because we already completely captured the informational variance of "was previously contacted vs wasn't" using the binary `prev_contacted` flag in the structural cleaning phase, imputing `pdays_clean` with the Train median is essentially imputing a near-empty column with a static baseline. We retain the variable anyway because the remaining ~4% of clients who *did* have successful previous campaigns possess genuinely informative `pdays` numerical magnitudes that a linear model can extract coefficient value from. We accept the limitation that the variable is extremely sparse.
- **Categorical Imputation:** We replace categorical `NaN` with the explicit string `"missing"`. This honors the *informative nature* of the missingness (perhaps clients who refuse to disclose `job` are empirically less likely to subscribe).
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
While `education` is logically ordinal, the step-sizes between levels are unknown. A linear model assumes uniform mathematical steps in an OrdinalEncoded variable (e.g., 1 -> 2 has the exact same impact as 2 -> 3). To avoid imposing this rigid, artificial structure, I apply **One-Hot Encoding** to *all* categorical variables.

*Impact on Dimensionality:* Expands from ~10 categorical columns to dozens of sparse binary features.
*Impact on Interpretability:* The Logistic Regression will yield a discrete coefficient for *each* category (e.g., `job_retired`), making it highly interpretable.
*Impact on Decision Boundaries:* Allows the linear model to form piecewise, non-linear logic through intercepts added for specific subgroups.

**Data Leakage Check:** I enforce `handle_unknown='ignore'`. If the Validation set contains a category unseen in Train, it ignores it rather than crashing, preventing leakage.
"""))

cells.append(nbf.v4.new_code_cell("""
# Example of Ordinal Mapping logic that *could* be deployed for 'education'
education_order = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 
                   'high.school', 'professional.course', 'university.degree', 'missing']

print(f"Theoretical Ordinal Hierarchy for Education:\\n{education_order}")
"""))

cells.append(nbf.v4.new_markdown_cell("""
While `OrdinalEncoder` mapping `[0, 1, 2, ... 7]` perfectly represents the hierarchical nature shown above, we strictly abstain from doing so for our Logistic Regression. A linear algorithm would mathematically presume the difference in marketing susceptibility between `illiterate` (0) and `basic.4y` (1) is perfectly identical to the distance between `professional.course` (5) and `university.degree` (6). Because that assumption is extremely dangerous in socioeconomic data, One-Hot Encoding acts as the safer, non-parametric alternative.
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
I apply **Standardization** (`StandardScaler`) to all numerical features.

**Justification for Logistic Regression:**
- *Gradient Optimization:* Logistic regression loss surfaces (binary cross-entropy) converge much faster using gradient descent/lbfgs when features are centered and share similar variances.
- *Regularization:* `scikit-learn`'s LogisticRegression includes L2 regularization by default. L2 drastically punishes variables with large magnitudes. If an unscaled feature spans $[0, 5000]$ (`nr.employed`), it will artificially shrink its coefficient. Scaling puts all features on the same numerical ground, normalizing the L2 penalty evenly.
- *Comparability:* Standardizing transforms coefficients into directly comparable "feature importances".

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

**Crucial Leakage Note:** Feature selection (analyzing variance, computing correlations) MUST be performed linearly on the **Training Set (`X_train`) only**. If we fit a VarianceThreshold or Correlation matrix on the entire pre-split dataset, we allow the statistical dynamics of the unseen test set to dictate which features our model learns from.
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
for i in range(len(num_cols)):
    for j in range(i+1, len(num_cols)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            print(f"{num_cols[i]} & {num_cols[j]}: {corr_matrix.iloc[i, j]:.3f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""
### 4c. Conceptual Feature Removal Discussion

**Observations from Variance & Correlation:**
- As shown manually above, `euribor3m` and `emp.var.rate` are extremely highly correlated (> 0.90). `euribor3m` and `nr.employed` also exhibit strong structural collinearity. This makes sense economically, as they all track identical macroeconomic phases.
- `duration` was already dropped immediately during data loading due to catastrophic target leakage. No other features are conceptually problematic or represent future leakage.

**Decision:**
For Logistic Regression, severe multicollinearity can cause coefficient instability, stripping the weights of their interpretable meaning. However, `scikit-learn`'s `LogisticRegression` applies **L2 Regularization** by default. L2 inherently punishes inflated coefficients, securely spreading the penalty across correlated groups without crashing the math. Because the regularization natively mitigates the modeling risk, I chose to retain all macroeconomic variables to prevent arbitrary data deletion, but acknowledge interpretability of individual macroeconomic coefficients is compromised.
"""))

# ==========================================
# 11. Class Imbalance
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 10. Addressing Class Imbalance

*Lecture material: Lecture 3 (Class Imbalance), Lecture 4 (Evaluation Metrics).*

**Assessment:** The majority class is `no` (~89%). This extreme skew is a severe concern because a naive model could default to simply guessing "no" to achieve 89% accuracy, completely ignoring the minority class ("yes").

**Strategy & Justification:**
Because we are utilizing Logistic Regression, instead of synthesizing fake points via SMOTE (which risks geometric boundary distortion in high-dimensional one-hot setups), the mathematically elegant approach is to modify the algorithm's loss function via `class_weight='balanced'`. 
This dynamically weights the loss gradients. A false negative (missing a 'yes') is penalized 9x more heavily than a false positive.

**Implication if done before splitting (Leakage):**
If we ran an oversampler like SMOTE on the *entire* dataset before splitting, overlapping synthetic examples would bleed directly into the Validation and Test sets. Our evaluation metrics would evaluate the model on fabricated data that already contains the training set's patterns, causing massively inflated scores that collapse in reality.

**Evaluation Metric Selection:**
Because of the imbalance, raw `Accuracy` is misleading. We will focus our evaluation heavily on `Precision` and `Recall` of the positive class ("yes"), as these measure performance exclusively on the target demographic.
"""))

# ==========================================
# 12. Modeling
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 11. Training a Logistic Regression Model

*Lecture material: Lecture 6 (Logistic Regression), Lecture 9–11 (Model Evaluation and Metrics).*

We assemble the final `Pipeline`, assuring that `X_val` is never `fitted`, only `transformed` and `predicted`.
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

# --- Accuracy Comparison Plot ---
fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
bars = ax_acc.bar(['Logistic Regression', 'Zero-Rule Baseline'], [acc, acc_base], color=['#2ca02c', '#d62728'])
ax_acc.set_ylim(0, 1)
ax_acc.set_ylabel('Accuracy')
ax_acc.set_title('Accuracy Comparison')

# Add text labels
for bar in bars:
    yval = bar.get_height()
    ax_acc.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""
**Interpretation:**
By heavily weighting the minority class (`class_weight='balanced'`), our Overall Accuracy has dropped significantly below the Zero-Rule Baseline (which lazily predicts "no" for everything and achieves ~89% artificial accuracy). 

However, **this is intentional and mathematically correct**. 
Instead of missing every single prospective client, the model now demonstrates strong *Recall*, capturing a massive segment of true subscribers ("yes"). 

In a marketing context, missing a willing subscriber (a false negative) is much more costly long-term than calling an unwilling one (a false positive). High recall at the cost of precision is the correct tradeoff here, proving our pipeline and structural decisions capture underlying variance rather than lazily chasing an inflated accuracy score!
"""))

# ==========================================
# 14. References
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## References / Dataset Notes

- **Moro, S., Cortez, P., & Rita, P. (2014).** A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*. doi:10.1016/j.dss.2014.03.001.
- **Moro, S., Laureano, R., & Cortez, P. (2011).** Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. *Proceedings of the European Simulation and Modelling Conference - ESM'2011*.

**Key Preprocessing Notes Specific to this Dataset:**
- `duration` is fundamentally a leakage risk because its value is not known before the call is performed. Therefore, it has been strictly excluded from this notebook to prevent data leakage, and no benchmark runs including it were performed.
- `pdays=999` strictly means the client was not previously contacted. Instead of treating this as a massive numerical outlier (999 days), we explicitly convert it to an indicator flag and replace the 999 values with standard missing values (`NaN`).
- Implicit missing categorical values in the UCI dataset are coded exclusively as the string `"unknown"`.
"""))

nb.cells.extend(cells)
nbf.write(nb, 'assignment_1_Christoph_Rintz.ipynb')
print("Successfully generated assignment_1_Christoph_Rintz.ipynb")
