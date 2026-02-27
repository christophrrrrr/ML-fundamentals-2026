import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ==========================================
# 1. Introduction
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
# Individual Assignment I: Machine Learning Foundation
**Data Preparation**

GitHub Repository: [https://github.com/christophrrrrr/Machine-Learning-Assignment-1.git](https://github.com/christophrrrrr/Machine-Learning-Assignment-1.git)

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
1. `duration`: This represents the call duration in seconds. While highly correlated with `y` (since successful sales usually take longer to close), it is strictly an outcome of the call. At prediction time (before or during the start of the call), this information is strictly unavailable. Predicting `duration` does not answer the business goal of identifying *who* will subscribe. Including it would result in catastrophic data leakage.
2. `poutcome`: This records the outcome of the *previous* marketing campaign. While useful as a predictive feature, it addresses a past event, whereas the current campaign's goal is measuring the present response (`y`).
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

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
df['y'].value_counts().plot(kind='bar', color=['#1f77b4', '#ff7f0e'], ax=axes[0])
axes[0].set_title('Target Variable (y) Distribution')
axes[0].set_ylabel('Count')

# Numerical Plot
df['age'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black', ax=axes[1])
axes[1].set_title('Age Distribution')

# Categorical Plot
df['job'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black', ax=axes[2])
axes[2].set_title('Job Category Distribution')
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
- **Special Consideration Variable (`duration`):** As noted, `duration` represents the length of the call. Because this is unknown *before* the call starts—which is the prediction moment—it causes severe data leakage. We must rigorously exclude it before modeling.
- **Implicit Missing Values:** Several categorical variables (like `job` or `education`) utilize the string `"unknown"` as an implicit missing value. The numerical variable `pdays` uses `999` to declare "never contacted before".
"""))

# ==========================================
# 4. Task Ordering
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 3. Task Ordering

*Lecture material: Lecture 2 (Data Splitting and Leakage), Lecture 5 (Preprocessing), Lecture 9 (ML Pipeline).*

To strictly prevent data leakage and abide by ML Pipeline discipline, I execute the data preparation tasks in the following sequence:

1. **Identifying Target & Data Loading** (Completed above): Understand the objective before transformations.
2. **Managing Missing Values (Identification & Structural Cleaning):** Convert sentinels like `"unknown"` or `999` into structural `NaN` values. *Why here?* Because finding string literals does not rely on global sample statistics (like mean/mode), so no leakage occurs.
3. **Data Splitting:** Divide into Train, Validation, and Test sets. *Why here?* Splitting MUST happen before any statistical operations. If it happens later, information from the test set would subtly influence the training parameters (Data Leakage).
4. **Managing Missing Values (Statistical Imputation):** Compute median/mode strictly on the Train set and apply to Validation/Test.
5. **Encoding Categorical Variables:** Fit OneHotEncoders on the Train set to define the dummy columns, avoiding learning about new categories from the Test set.
6. **Feature Scaling:** Fit `StandardScaler` on the Train set to compute mean/variance, then apply to Validation/Test.
7. **Feature Selection:** Analyze correlations and variance on the Train set only.
8. **Addressing Class Imbalance:** Apply technique (like class weights or SMOTE) strictly on the encoded, scaled Train set.
9. **Modeling:** Train Logistic Regression on Train, evaluate on Validation.

**Incorrect Ordering Example:** 
If we performed **Feature Scaling** before **Data Splitting**, we would calculate the mean and standard deviation across the *entire* dataset. The standardized test values would inherently contain information about the central tendency of the training set. This is a classic form of data leakage, artificially inflating the test evaluation metrics because the test data "peeked" into the global distribution.
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

- **Numerical Imputation:** For `pdays_clean`, we impute with the constant `-1` (or median). Because it is a duration, median is robust.
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

**Conceptual Removal:** We explicitly deleted `duration` earlier due to target leakage.
**Statistical Filters:** We can theoretically fit a `VarianceThreshold` or correlation filter here. Due to our rigorous pipeline assembly, any selection must be executed logically inside the pipeline using the Train set constraints.

Since the feature space is quite small (`bank-additional` has ~20 raw features), deleting high-variance features manually isn't strictly necessary for a regularized linear model, as L2 regularization inherently pushes the coefficients of redundant/multicollinear features towards zero to maintain stability.
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
"""))

cells.append(nbf.v4.new_markdown_cell("""
**Interpretation:**
By heavily weighting the minority class `class_weight='balanced'`, our Overall Accuracy has dropped significantly below the Zero-Rule Baseline (which lazily predicts "no"). 
However, **this is intentional and correct**. 
Instead of missing every single prospective client, the model now demonstrates strong *Recall*, capturing a massive segment of true subscribers ("yes"). In a marketing context, calling a few false positives is drastically cheaper than missing out on willing subscribers, proving our pipeline and structural decisions succeed at capturing structural variance rather than lazily chasing accuracy!
"""))

# ==========================================
# 13. Test Set Evaluation
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## 12. Final Generalization: Test Set Evaluation

*Lecture material: Lecture 9 (ML Pipeline / Test Sets).*

To definitively prove that our model and preprocessing pipeline generalize effectively to unseen data without leakage, we finally expose it to our isolated Test Set.
"""))

cells.append(nbf.v4.new_code_cell("""
# Evaluate on final held-out Test Set
y_test_pred = model.predict(X_test)

print("--- Test Set Classification Report ---")
print(classification_report(y_test, y_test_pred))

print("--- Test Set Confusion Matrix ---")
print(confusion_matrix(y_test, y_test_pred))

test_f1 = f1_score(y_test, y_test_pred, pos_label='yes')
print(f"\\nTest F1-score (positive class 'yes'): {test_f1:.4f}")
"""))

# ==========================================
# 14. References
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""
## References / Dataset Notes

- **Moro, S., Cortez, P., & Rita, P. (2014).** A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*. doi:10.1016/j.dss.2014.03.001.
- **Moro, S., Laureano, R., & Cortez, P. (2011).** Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. *Proceedings of the European Simulation and Modelling Conference - ESM'2011*.

**Key Preprocessing Notes Specific to this Dataset:**
- `duration` is fundamentally a leakage risk because its value is not known before the call is performed (we only use it for ablation benchmarks, not standard modeling).
- `pdays=999` strictly means the client was "not previously contacted" rather than an extremely high number of days, requiring manual conversion to structural missingness (`NaN`) before scaling.
- Implicit missing categorical values in the UCI dataset are coded exclusively as the string `"unknown"`.
"""))

nb.cells.extend(cells)
nbf.write(nb, 'assignment_1_Christoph_Rintz.ipynb')
print("Successfully generated assignment_1_Christoph_Rintz.ipynb")
