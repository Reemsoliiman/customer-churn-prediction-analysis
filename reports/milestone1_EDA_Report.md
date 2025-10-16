# Exploratory Data Analysis Report: Customer Churn Dataset

## 1. Introduction

This report presents the exploratory data analysis (EDA), preprocessing, and key findings for a customer churn dataset as part of Milestone 1: Data Collection, Exploration, and Preprocessing. The objectives were to collect a dataset with customer demographics, usage patterns, and subscription details; explore its structure and relationships; preprocess the data for machine learning; and visualize patterns relevant to churn prediction. The dataset, sourced from the merged `churn-bigml-20.csv` and `churn-bigml-80.csv` files (processed into `merged_churn_data.csv`), contains 3333 rows and 20 columns, providing a robust foundation for churn analysis.

## 2. Dataset Overview

The dataset includes features critical for churn prediction, categorized as follows:

- **Customer Demographics**: State (object), Area code (integer, treated as categorical).
- **Usage Patterns**: Total day minutes, Total day calls, Total eve minutes, Total eve calls, Total night minutes, Total night calls, Total intl minutes, Total intl calls, Number vmail messages, Customer service calls (numerical).
- **Subscription Details**: International plan, Voice mail plan (object), Total day charge, Total eve charge, Total night charge, Total intl charge (numerical).
- **Other**: Account length (numerical, tenure-related).
- **Target**: Churn (boolean, True/False).

The dataset has 3333 rows and 20 columns, with no missing values or duplicates initially identified.

## 3. Data Exploration

### 3.1 Dataset Structure
The dataset comprises:
- 3 object columns (State, International plan, Voice mail plan).
- 1 integer column (Area code, treated as categorical).
- 8 integer columns (e.g., Total day calls, Customer service calls).
- 8 float columns (e.g., Total day minutes, Total day charge).
- 1 boolean column (Churn).

### 3.2 Numerical Distributions
Key statistics from numerical features include:
- Account length: Mean ~100.62 months, max 243, indicating potential outliers in tenure.
- Total day minutes: Mean ~179.48, max 350.8, suggesting high usage outliers.
- Customer service calls: Mean ~1.56, max 9, with discrete values.
- Number vmail messages: Mean ~8.02, min 0, max 50, highly skewed with many zeros.

These distributions highlight skewness and potential outliers in usage and tenure features.

### 3.3 Churn Distribution
The Churn column shows ~14% of customers churned (True), indicating a class imbalance that may require techniques like oversampling or class weighting during modeling.

## 4. Preprocessing and Feature Engineering
Preprocessing steps were applied to prepare the dataset:
- **Missing Values and Duplicates**: No missing values or duplicates found initially.
- **Outlier Handling**: Continuous features (e.g., Total day minutes) were clipped using the IQR method to remove extreme values.
- **Interim Data**: Saved as `imputed_data.csv` and `outlier_removed_data.csv` in the `data/interim/` folder.

## 5. EDA Visualizations
Visualizations were created to identify patterns:
- **Scatter Plots**: Three scatter plots (`Customer service calls vs Total_Charge`, `Total day minutes vs Total_Minutes`, `Number vmail messages vs Total intl calls`) by Churn were generated. Key observations include:
  - Higher `Customer service calls` and `Total_Charge` values show a slight clustering of churned customers.
  - `Total day minutes` and `Total_Minutes` exhibit a spread with potential outliers among churned users.
  - `Number vmail messages` vs `Total intl calls` shows minimal differentiation, suggesting weaker correlation with churn.
- These plots are saved as `scatter_plots_key_features.png` in `visualizations/enhanced/`.

## 6. Key Findings
- **Churn Imbalance**: Approximately 14% of customers churned, indicating a need for class imbalance handling.
- **Usage Patterns**: Higher `Customer service calls` and potential outliers in `Total day minutes` are associated with churn.
- **Visual Patterns**: Scatter plots reveal initial trends, with further analysis needed to confirm relationships.

## 7. Conclusion
The EDA and preprocessing of the customer churn dataset successfully met the objectives of Milestone 1. The dataset was explored for structure and distributions, preprocessed with outlier handling, and visualized to identify initial patterns. The preprocessed dataset and interim files are ready for further analysis in subsequent milestones.