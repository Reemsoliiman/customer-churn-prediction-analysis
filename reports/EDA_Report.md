# Exploratory Data Analysis (EDA) Report

## Executive Summary

This report summarizes key insights from the exploratory data analysis performed on the Telecom Customer Churn dataset. The analysis was conducted using the merged dataset (churn-bigml-20.csv + churn-bigml-80.csv) containing 3,333 customer records with 20 features.

---

## Dataset Overview

- **Total Records**: 3,333 customers
- **Total Features**: 20 (19 features + 1 target variable)
- **Churn Rate**: 14.49% (imbalanced dataset)
- **Data Quality**: Clean dataset with minimal missing values after preprocessing

*Note: Exact churn/retained counts may vary slightly based on data preprocessing steps.*

---

## Key Findings

### 1. Target Variable Distribution

- **Churn Rate**: 14.49% indicates an imbalanced dataset
- This imbalance requires special handling during model training (SMOTE oversampling implemented)
- The churn rate is typical for telecom industry standards

### 2. Customer Demographics

#### Account Length
- Average account length varies between churned and retained customers
- Newer customers (first year) show higher churn risk
- Account length distribution shows customers with very short tenure (< 30 days) have elevated churn rates

#### Geographic Distribution
- State-level analysis performed (one-hot encoded for model training)
- Area code feature removed as non-predictive

### 3. Service Plan Analysis

#### International Plan
- **Strong churn indicator**: Customers with international plans show significantly higher churn rates
- Visualizations reveal clear separation between churners and non-churners based on international plan status

#### Voice Mail Plan
- Customers with voice mail plans show lower churn rates
- Voice mail usage (number of messages) correlates with customer retention

### 4. Usage Patterns

#### Call Minutes Distribution
- **Day Minutes**: Churned customers show different usage patterns compared to retained customers
- **Evening Minutes**: Similar patterns observed
- **Night Minutes**: Usage patterns differ between churners and non-churners
- **International Minutes**: Higher usage in churned customers

#### Call Frequency
- Total call counts show variations between churned and retained customers
- Day, evening, and night call patterns analyzed

### 5. Customer Service Interactions

- **Critical Finding**: Customer service calls is one of the strongest predictors of churn
- Customers with 3+ service calls show dramatically higher churn rates
- Service call distribution clearly separates churners from non-churners
- This feature emerged as a key risk indicator in segmentation analysis

### 6. Data Quality Issues Identified

#### Missing Values
- Minimal missing values detected
- Handled through median imputation (numerical) and mode imputation (categorical)

#### Duplicates
- Duplicate records removed during preprocessing

#### Outliers
- Outliers detected in usage metrics (minutes, calls)
- Handled using IQR-based clipping method

---

## Visualizations Created

The EDA process generated multiple visualizations saved in `visualizations/`:

### Static Visualizations (`visualizations/static/`)
- Churn distribution charts
- Account length density plots
- Service calls histograms
- Usage pattern distributions (day, evening, night, international)
- Categorical churn rate comparisons
- Correlation heatmaps
- Pair plots

### Key Insights from Visualizations

1. **Service Calls vs Churn**: Clear positive correlation - more service calls = higher churn risk
2. **International Plan**: Strong association with churn (higher churn rate)
3. **Usage Patterns**: Extreme users (very high or very low usage) show higher churn rates
4. **Account Age**: First-year customers are at higher risk

---

## Preprocessing Decisions

### 1. Feature Removal
- **Charge columns**: Removed as redundant (perfectly correlated with minutes)
- **Area code**: Removed as non-predictive

### 2. Encoding Strategy
- **Binary features** (International plan, Voice mail plan): Encoded as 1/0
- **State**: One-hot encoded (drop_first=True to avoid multicollinearity)
- **Churn**: Encoded as 1/0

### 3. Missing Value Handling
- Numerical features: Median imputation
- Categorical features: Mode imputation

### 4. Outlier Treatment
- IQR-based clipping applied to numerical features
- Prevents extreme values from skewing model performance

---

## Data Distribution Characteristics

### Numerical Features
- Most usage metrics follow approximately normal distributions
- Some features show right-skewed distributions (log transforms applied during feature engineering)
- Account length shows relatively uniform distribution

### Categorical Features
- International plan: ~10% of customers have international plans
- Voice mail plan: ~30% of customers have voice mail plans
- State: Distributed across 51 US states

---

## Risk Segments Identified

### High-Risk Customer Profile
- 3+ customer service calls
- International plan holders
- Very high or very low usage patterns
- New customers (< 1 year tenure)

### Low-Risk Customer Profile
- 0-1 customer service calls
- Voice mail plan holders
- Moderate, consistent usage patterns
- Longer tenure (> 2 years)

---

## Recommendations from EDA

1. **Feature Engineering Priority**: Focus on customer service calls, international plan status, and usage pattern interactions
2. **Class Balancing**: Implement SMOTE or similar technique due to 14.49% churn rate
3. **Feature Selection**: Use RFE to identify most predictive features (implemented: top 20 features selected)
4. **Monitoring**: Track service call frequency as early warning indicator

---

## Next Steps

The EDA findings informed:
- Feature engineering strategy (see `docs/Feature_Engineering_Summary.md`)
- Model selection approach
- Preprocessing pipeline design
- Risk segmentation framework

---

**Report Generated From**: `notebooks/01_data_exploration.ipynb`  
**Date**: Based on analysis in project notebooks  
**Dataset**: Kaggle Telecom Customer Churn Dataset (merged 20% + 80% splits)

