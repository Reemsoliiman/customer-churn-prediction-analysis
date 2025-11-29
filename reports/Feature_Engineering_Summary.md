# Feature Engineering Summary

## Overview

This document outlines all feature engineering transformations applied to the telecom churn dataset. All transformations are implemented in `src/utils/helpers.py` and are used consistently across both training and inference pipelines.

---

## Feature Engineering Pipeline

The feature engineering process is implemented in the `engineer_features()` function, which applies transformations in a specific order to ensure consistency.

---

## 1. Feature Removal

### Charge Columns
- **Action**: All columns containing "charge" in the name are dropped
- **Rationale**: Charge columns are perfectly correlated with minutes (charge = minutes × rate)
- **Impact**: Reduces multicollinearity and feature redundancy
- **Code**: `charge_cols = [c for c in df.columns if "charge" in c.lower()]`

---

## 2. Derived Features

### Customer Tenure
- **Feature Name**: `Customer_tenure_months`
- **Transformation**: `Account length / 30.0`
- **Purpose**: Convert account length from days to months for better interpretability
- **Use Case**: Enables monthly-based usage pattern calculations

### Total Aggregates

#### Total Minutes
- **Feature Name**: `Total_Minutes`
- **Transformation**: Sum of all minute columns
  - Total day minutes
  - Total eve minutes
  - Total night minutes
  - Total intl minutes
- **Purpose**: Capture overall usage volume

#### Total Calls
- **Feature Name**: `Total_Calls`
- **Transformation**: Sum of all call columns
  - Total day calls
  - Total eve calls
  - Total night calls
  - Total intl calls
- **Purpose**: Capture overall call frequency

---

## 3. Usage Pattern Features

### Average Daily Usage
- **Feature Name**: `Avg_Daily_Usage`
- **Transformation**: `Total_Minutes / (Customer_tenure_months + 1)`
- **Purpose**: Normalize usage by account age to identify usage intensity
- **Interpretation**: Higher values indicate heavy users relative to tenure

### Call Frequency
- **Feature Name**: `Call_Frequency`
- **Transformation**: `Total_Calls / (Customer_tenure_months + 1)`
- **Purpose**: Normalize call count by account age
- **Interpretation**: Measures calling activity rate

### International Usage Rate
- **Feature Name**: `Intl_Usage_Rate`
- **Transformation**: `Total intl minutes / (Total_Minutes + 1)`
- **Purpose**: Proportion of international usage relative to total usage
- **Interpretation**: Higher values indicate international-focused customers

---

## 4. Behavioral Flags

### High Service Calls
- **Feature Name**: `High_Service_Calls`
- **Transformation**: Binary flag (1 if Customer service calls > 3, else 0)
- **Purpose**: Identify customers with elevated service call frequency
- **Rationale**: Statistical analysis showed 3+ calls as critical threshold
- **Impact**: Strong predictor of churn risk

### Voice Mail Usage
- **Feature Name**: `Has_Vmail`
- **Transformation**: Binary flag (1 if Number vmail messages > 0, else 0)
- **Purpose**: Identify customers actively using voice mail
- **Rationale**: Voice mail usage correlates with retention

---

## 5. Log Transformations

### Applied to:
- `Total_Minutes`
- `Total_Calls`
- `Avg_Daily_Usage`
- `Call_Frequency`

### Transformation
- **Method**: `np.log1p()` (log(1 + x))
- **Purpose**: 
  - Handle right-skewed distributions
  - Reduce impact of extreme outliers
  - Improve model performance on skewed features
- **Safety**: Features are clipped to non-negative values before transformation

### Generated Features
- `log_Total_Minutes`
- `log_Total_Calls`
- `log_Avg_Daily_Usage`
- `log_Call_Frequency`

---

## 6. Interaction Features

### Usage-Service Interaction
- **Feature Name**: `Usage_Service_Interaction`
- **Transformation**: `(Total_Minutes / max(Total_Minutes)) × Customer service calls`
- **Purpose**: Capture combined effect of high usage and service issues
- **Interpretation**: Identifies high-value customers with service problems (at-risk segment)
- **Rationale**: Statistical analysis showed interaction between usage and service calls

### International Plan Usage Flag
- **Feature Name**: `Intl_Plan_Usage_Flag`
- **Transformation**: Binary flag
  - 1 if (International plan == 1) AND (Total intl minutes > median)
  - 0 otherwise
- **Purpose**: Identify international plan holders with high international usage
- **Interpretation**: Flags customers actively using international features

### Day Usage Ratio
- **Feature Name**: `Day_Usage_Ratio`
- **Transformation**: `Total day minutes / (Total_Minutes + 1)`
- **Purpose**: Proportion of daytime usage
- **Interpretation**: Higher values indicate business customers (daytime-focused)

### Tenure-Usage Score
- **Feature Name**: `Tenure_Usage_Score`
- **Transformation**: `Customer_tenure_months × Avg_Daily_Usage`
- **Purpose**: Identify loyal, high-value customers
- **Interpretation**: High score = long-tenured, high-usage customers (retention segment)

### Service Calls Per Month
- **Feature Name**: `Service_Calls_Per_Month`
- **Transformation**: `Customer service calls / (Customer_tenure_months + 1)`
- **Purpose**: Normalize service calls by account age
- **Interpretation**: Identifies "chronic complainers" (frequent service issues relative to tenure)

---

## 7. Categorical Encoding

### Binary Features
- **International plan**: "Yes"/"No" → 1/0
- **Voice mail plan**: "Yes"/"No" → 1/0
- **Churn**: Boolean/string → 1/0

### One-Hot Encoding
- **State**: One-hot encoded with `drop_first=True` to avoid multicollinearity
- **Prefix**: "State_" added to each state column
- **Result**: 50 binary features (51 states - 1 dropped)

### Feature Removal
- **Area code**: Dropped (non-predictive, redundant with State)

---

## 8. Feature Selection

### Recursive Feature Elimination (RFE)
- **Method**: RFE with RandomForestClassifier (200 estimators)
- **Features Selected**: Top 20 features
- **Implementation**: `src/pipelines/engineer_features.py`
- **Output**: `models/artifacts/selected_features.pkl`

### Selection Criteria
- Feature importance from RandomForest
- Recursive elimination of least important features
- Final set optimized for model performance

---

## Feature Engineering Statistics

### Input Features
- **Raw features**: ~20 features (after basic cleaning)
- **After encoding**: ~70 features (including one-hot encoded states)

### Engineered Features Created
- **Derived features**: 2 (Total_Minutes, Total_Calls)
- **Usage patterns**: 3 (Avg_Daily_Usage, Call_Frequency, Intl_Usage_Rate)
- **Behavioral flags**: 2 (High_Service_Calls, Has_Vmail)
- **Log transforms**: 4 (log_Total_Minutes, log_Total_Calls, log_Avg_Daily_Usage, log_Call_Frequency)
- **Interaction features**: 5 (Usage_Service_Interaction, Intl_Plan_Usage_Flag, Day_Usage_Ratio, Tenure_Usage_Score, Service_Calls_Per_Month)
- **Customer tenure**: 1 (Customer_tenure_months)

**Total engineered features**: ~17 new features

### Final Feature Set
- **After RFE**: 20 selected features
- **Used in model**: Top 20 features from RFE

---

## Feature Importance Insights

Based on statistical analysis and model performance:

### Top Predictive Features
1. **Customer service calls** (original + derived features)
2. **International plan** (original + interaction features)
3. **Usage patterns** (Total_Minutes, Avg_Daily_Usage)
4. **Voice mail features** (Has_Vmail, Number vmail messages)
5. **Interaction features** (Usage_Service_Interaction, Tenure_Usage_Score)

---

## Implementation Details

### Code Location
- **Main function**: `src/utils/helpers.py::engineer_features()`
- **Pipeline integration**: `src/pipelines/engineer_features.py`
- **Inference usage**: `src/api/predict.py` (via `engineer_features()`)

### Consistency Guarantees
- Same function used in training and inference
- Feature alignment via `align_features_for_prediction()`
- Selected features stored in `selected_features.pkl`

### Data Safety
- All transformations handle missing values gracefully
- Log transforms use `log1p()` to handle zeros
- Division operations include `+ 1` to prevent division by zero
- Features checked for existence before transformation

---

## Expected Impact on Model Performance

### Benefits
1. **Improved predictive power**: Interaction features capture non-linear relationships
2. **Better handling of skewed data**: Log transforms normalize distributions
3. **Reduced overfitting**: RFE selects most relevant features
4. **Interpretability**: Derived features have clear business meaning

### Feature Engineering Impact
- **Before engineering**: Basic features only
- **After engineering**: Rich feature set with interactions and patterns
- **Model improvement**: Significant performance gains observed (see Model Evaluation Report)

---

## Maintenance Notes

### Adding New Features
1. Add transformation logic to `engineer_features()` function
2. Ensure consistency between training and inference
3. Update feature selection if needed
4. Test with both training data and single predictions

### Modifying Existing Features
1. Update `engineer_features()` function
2. Retrain models (feature changes require retraining)
3. Update `selected_features.pkl` if RFE is re-run
4. Verify inference pipeline still works

---

**Documentation Source**: `src/utils/helpers.py`  
**Implementation**: `src/pipelines/engineer_features.py`  
**Feature Selection**: RFE with RandomForestClassifier (top 20 features)

