# 📊 Customer Churn Analysis  
Data Preprocessing & Exploratory Data Analysis (EDA)  
Machine Learning & Data Science – ENCS5341  

---

## 📌 Project Overview  
This project performs preprocessing and exploratory data analysis on a synthetic telecom-like customer dataset.  
The goal is to clean the dataset, discover insights, visualize patterns, and prepare for predictive modeling of customer churn.  

This work fulfills the requirements of Assignment #1 for ENCS5341 (Machine Learning & Data Science).

---

## 📁 Dataset Description  
| Feature | Type | Description |
|--------|------|-------------|
| CustomerID | Numeric | Unique customer identifier |
| Age | Numeric | Customer age |
| Gender | Categorical | 0: Male, 1: Female |
| Income | Numeric | Annual income in USD |
| Tenure | Numeric | Years with company |
| ProductType | Categorical | 0: Basic, 1: Premium |
| SupportCalls | Numeric | Number of support calls in the last year |
| ChurnStatus | Binary | 1: Churned, 0: Stayed |

---

## 🧹 Data Preprocessing  

### 1️⃣ Handling Missing Data
- Income and SupportCalls had missing values → filled with **median** due to skewed distributions.
- Tenure and Gender contained **no missing values**.
- Age invalid records (<18) removed.

### 2️⃣ Handling Outliers
- Outliers detected using **IQR**.
- Income extreme values capped within valid bounds.
- SupportCalls extreme values replaced with median.
- Tenure had no significant outliers.

### 3️⃣ Feature Scaling
Standardization applied to numerical features:
