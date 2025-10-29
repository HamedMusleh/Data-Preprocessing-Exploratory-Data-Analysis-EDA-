# ========================== first ==========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('customer_data.csv') 
print("Dataset loaded successfully.")

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info() 

print("\nSummary Statistics:")
print(df.describe()) 


# ========================== Second ==========================
print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values)

missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nMissing values percentage per column:")
print(missing_percentage)

df_before_impute = df.copy()

# ========================== Imputation ==========================
print("\nImputing missing values...")

mean_age = df['Age'].mean()
df['Age'].fillna(mean_age, inplace=True)
print(f"Imputed 'Age' with mean: {mean_age:.2f}")

median_income = df['Income'].median()
df['Income'].fillna(median_income, inplace=True)
print(f"Imputed 'Income' with median: {median_income:.2f}")

mean_tenure = df['Tenure'].mean()
df['Tenure'].fillna(mean_tenure, inplace=True)
print(f"Imputed 'Tenure' with mean: {mean_tenure:.2f}")

median_support_calls = df['SupportCalls'].median()
df['SupportCalls'].fillna(median_support_calls, inplace=True)
print(f"Imputed 'SupportCalls' with median: {median_support_calls:.2f}")

print("\nMissing values after imputation:")
print(df.isnull().sum())

print("\nDataset Info after imputation:")
df.info()


# ========================== Median Justification Block ==========================

features_to_check = ['Income', 'SupportCalls', 'Age', 'Tenure']

def outlier_counts(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    n_out = ((series < lb) | (series > ub)).sum()
    return Q1, Q3, IQR, lb, ub, int(n_out)

print("\n==================== Why Mean/Median? Evidence (pre-imputation data) ====================")
for col in features_to_check:
    s = df_before_impute[col].dropna()

    # Histogram مع mean/median (Pre-Imputation)
    plt.figure(figsize=(7,4))
    sns.histplot(s, kde=True, bins=30)
    plt.axvline(s.mean(), linestyle='--', label=f"Mean = {s.mean():.2f}")
    plt.axvline(s.median(), linestyle='-', label=f"Median = {s.median():.2f}")
    plt.title(f'{col} Distribution (Pre-Imputation)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Boxplot (Pre-Imputation)
    plt.figure(figsize=(5,4))
    sns.boxplot(y=s)
    plt.title(f'{col} Box Plot (Pre-Imputation)')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


    Q1, Q3, IQR, lb, ub, n_out = outlier_counts(s)
    stats_table = pd.DataFrame({
        'Statistic': ['Mean','Median','Std','Skewness','Min','Q1','Q3','Max','IQR','LowerBound','UpperBound','Outliers'],
        'Value': [s.mean(), s.median(), s.std(), s.skew(), s.min(), Q1, Q3, s.max(), IQR, lb, ub, n_out]
    })
    print(f"\n{col} Statistical Summary (Pre-Imputation):")
    print(stats_table.to_string(index=False))

    skew = s.skew()
    mean_median_gap = abs(s.mean() - s.median())
    gap_rel = mean_median_gap / (s.std() if s.std() != 0 else 1)
    use_median = (skew > 0.5 or skew < -0.5) or (gap_rel > 0.3) or (n_out > 0)

    print(f"\nDecision for {col}:")
    print(f"  Skewness = {skew:.3f}, |Mean - Median| = {mean_median_gap:.3f} (relative={gap_rel:.3f}), Outliers = {n_out}")
    if use_median:
        print("  → Recommendation: Use MEDIAN for imputation (robust to skewness/outliers).")
    else:
        print("  → Recommendation: Use MEAN for imputation (distribution fairly symmetric).")



# ========================== Third ==========================
numerical_features = ['Age', 'Income', 'Tenure', 'SupportCalls']

print("\nGenerating Box Plots for outliers")
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_features):
    plt.subplot(1, len(numerical_features), i + 1)
    sns.boxplot(y=df[col]) 
    plt.title(f'Box Plot for {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

from scipy import stats

print("\nIdentifying outliers using Z-score with threshold=3")
outliers_z = {}
for col in numerical_features:
    z_scores = np.abs(stats.zscore(df[col]))
    outlier_indices = np.where(z_scores > 3)[0]
    outliers_z[col] = len(outlier_indices)

print("Number of outliers detected by Z-score:")
print(outliers_z)

print("\nHandling outliers using Capping/Flooring which is based on IQR")
for col in ['Income', 'SupportCalls']:
    Q1 = df[col].quantile(0.25) 
    Q3 = df[col].quantile(0.75) 
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR

    print(f"\n{col}:")
    print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Lower Bound={lower_bound:.2f}, Upper Bound={upper_bound:.2f}")

    outliers_before = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"  Outliers found before capping: {outliers_before}")

    # Apply Capping
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    outliers_after = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"  Outliers found after capping: {outliers_after}")

# Verifying
print("\nGenerating Box Plots after handling outliers")
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_features):
    plt.subplot(1, len(numerical_features), i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot for {col} after Handling')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

print("\nSummary Statistics after handling outliers:")
print(df[numerical_features].describe())

# ========================== Fourth ==========================
from sklearn.preprocessing import StandardScaler

print("\nApplying Standardization (Z-score scaling)")
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nFirst 5 rows of scaled numerical data:")
print(df_scaled[numerical_features].head())

print("\nSummary Statistics of scaled numerical data:")
print(df_scaled[numerical_features].describe())

# ========================== Fifth (UPDATED) ==========================
print("\n==================== EDA: Univariate & Bivariate ====================")


# ========================== Distribution of Categorical Features ==========================
categorical_features = ['Gender', 'ProductType', 'ChurnStatus']
plt.figure(figsize=(10, 3))
for i, col in enumerate(categorical_features):
    plt.subplot(1, 3, i + 1)
    color_map = {0: '#6AB187', 1: 'lightcoral'}
    palette = [color_map[val] for val in sorted(df[col].unique())]

    sns.countplot(x=col, data=df, palette=palette)
    plt.title(f'{col} Distribution')

plt.tight_layout()
plt.show()



print("\n==================== EDA: Bivariate Analysis ====================")
# Numerical features vs Target (ChurnStatus)
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='ChurnStatus', y=col, data=df)
    plt.title(f'{col} vs ChurnStatus')
plt.tight_layout()
plt.show()

# Scatter plot example (Income vs Age colored by Churn)
plt.figure(figsize=(6, 5))
sns.scatterplot(x='Age', y='Income', hue='ChurnStatus', data=df)
plt.title('Age vs Income by ChurnStatus')
plt.show()

# Categorical vs Target (Bar plots showing churn rate per category)
cat_vars = ['Gender', 'ProductType']
for col in cat_vars:
    churn_rate = df.groupby(col)['ChurnStatus'].mean().reset_index()
    plt.figure(figsize=(5, 4))
    sns.barplot(x=col, y='ChurnStatus', data=churn_rate, palette='Set1')
    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Average Churn Rate')
    plt.show()

print("\n==================== EDA: Correlation Analysis ====================")


# Correlation matrix for numerical columns
corr_matrix = df[numerical_features + ['ChurnStatus']].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)


# Heatmap for correlation
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


