# 📊 Customer Churn Analysis  
Data Preprocessing & Exploratory Data Analysis (EDA)  
Machine Learning & Data Science – ENCS5341  

---

## 📌 Project Overview  
This project performs data preprocessing and exploratory data analysis on a synthetic telecom-like customer dataset.  
The purpose is to clean the data, visualize patterns, identify outliers, handle missing values, scale numerical features, and extract insights related to churn.  

This work fulfills the requirements of Assignment #1 for ENCS5341: Machine Learning and Data Science.

---

## 📁 Project Structure

- **Assignment_1.py** — Main script (preprocessing + EDA)
- **customer_data.csv** — Input dataset (required)
- **README.md** — Project documentation





---

## 🧩 Dataset Schema

| Feature        | Type        | Description |
|----------------|------------|-------------|
| CustomerID     | Numeric     | Unique identifier |
| Age            | Numeric     | Customer age |
| Gender         | Categorical | 0 = Male, 1 = Female |
| Income         | Numeric     | Annual income (USD) |
| Tenure         | Numeric     | Years with the company |
| ProductType    | Categorical | 0 = Basic, 1 = Premium |
| SupportCalls   | Numeric     | Support calls in the last year |
| ChurnStatus    | Binary      | 1 = Churned, 0 = Stayed |

---

## 🧹 Data Preprocessing Workflow

### 1️⃣ Missing Values Handling
- Income → median  
- SupportCalls → median  
- Age → mean  
- Tenure → mean  
- Justifications included using skewness/outliers evaluation before imputation  
- Age invalid entries (<18) removed

### 2️⃣ Outlier Detection & Treatment
- Initial check using boxplots and **Z-score** (|z| > 3)
- **IQR method** used for capping only on:
  - Income  
  - SupportCalls  
- Tenure had no significant outliers

### 3️⃣ Feature Scaling
Standardization (Z-score) applied to numerical features:
z = (x - μ) / σ

Scaled DataFrame saved internally for EDA usage.

---

## 🔍 EDA (Exploratory Data Analysis)

### Univariate Analysis
- Histograms + KDE + Boxplots for numerical features
- Bar charts for categorical distributions

### Bivariate Analysis
- ChurnStatus vs numerical features using boxplots
- Scatter plot: Age vs Income colored by churn

### Correlation Analysis
- Numeric correlation matrix
- Heatmap visualization

---

## 📈 Visualizations Produced
- Histogram + KDE + median/mean markers
- Numerical feature boxplots before/after handling outliers
- Countplots for categorical features
- Churn analysis vs ProductType and Gender
- Correlation heatmap

(All plots are displayed interactively during execution)

---

## ✅ Tools Used
- Python  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- SciPy  
- Scikit-learn  
- Jupyter or any Python execution environment  

---

## 🚀 How to Run

Install dependencies:
```bash
pip install pandas numpy seaborn matplotlib scipy scikit-learn
