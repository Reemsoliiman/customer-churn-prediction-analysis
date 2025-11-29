# Data Analysis Report: Statistical Tests and Feature Relationships

## Executive Summary

This report documents the comprehensive statistical analysis performed to understand feature relationships with customer churn. The analysis includes t-tests, ANOVA, chi-squared tests, correlation analysis, and effect size calculations to identify the most significant predictors of churn.

---

## Statistical Tests Performed

### 1. T-Tests for Numerical Features

**Method**: Independent samples t-test (Welch's t-test, unequal variances)

**Purpose**: Determine if numerical features show statistically significant differences between churned and non-churned customers.

**Results**:
- **13 out of 17 numerical features** are statistically significant (p < 0.05)
- All significant features show meaningful differences between churners and non-churners

#### Top Statistically Significant Features (by p-value):

1. **International plan** (p = 4.28e-23) - Highest significance
2. **Total day minutes** (p = 1.22e-20)
3. **Total day charge** (p = 1.22e-20) - Redundant with minutes
4. **Customer service calls** (p = 5.27e-18) - Critical predictor
5. **Voice mail plan** (p = 1.61e-11) - Retention indicator
6. **Number vmail messages** (p = 8.76e-09)
7. **Total eve minutes** (p = 1.84e-07)
8. **Total eve charge** (p = 1.84e-07)
9. **Total intl charge** (p = 9.03e-05)
10. **Total intl minutes** (p = 9.07e-05)
11. **Total intl calls** (p = 3.19e-03)
12. **Total night charge** (p = 3.03e-02)

#### Effect Size (Cohen's d)

Cohen's d was calculated to measure the practical significance of differences:

- **Positive Cohen's d** (Red markers): Higher values in churned customers
  - International plan, Total day minutes, Customer service calls, etc.
  
- **Negative Cohen's d** (Green markers): Lower values in churned customers
  - Voice mail plan, Number vmail messages

**Interpretation**: Features with large absolute Cohen's d values indicate strong practical significance beyond statistical significance.

---

### 2. Chi-Squared Tests for Categorical Features

**Method**: Chi-square test of independence with Cramér's V effect size

**Purpose**: Test association between categorical features and churn.

#### Results:

| Feature | Chi-square | p-value | Cramér's V | Interpretation |
|---------|------------|---------|------------|----------------|
| **International plan** | 222.57 | 2.49e-50 | 0.258 | **Strong association** |
| **Voice mail plan** | 34.13 | 5.15e-09 | 0.101 | Moderate association |

**Key Findings**:
- **International plan** shows the strongest categorical association with churn (Cramér's V = 0.258)
- **Voice mail plan** shows moderate association (Cramér's V = 0.101)
- Both features are highly statistically significant (p < 0.001)

**Cramér's V Interpretation**:
- 0.258 (International plan): Moderate to strong effect size
- 0.101 (Voice mail plan): Small to moderate effect size

---

### 3. ANOVA Tests

**Method**: One-way ANOVA (F-test)

**Purpose**: Test differences in churn rates across multiple groups.

#### Service Calls Groups ANOVA

**Grouping**: Customer service calls grouped into:
- Low: 0-1 calls
- Medium: 2-3 calls  
- High: 4+ calls

**Results**:
- **F-statistic**: 181.65
- **p-value**: 1.25e-74 (highly significant)
- **Conclusion**: Strong evidence that churn rates differ significantly across service call groups

**Interpretation**: Service call frequency is a critical segmentation variable for churn prediction.

---

## Correlation Analysis

### Feature Correlations with Churn

Correlation matrices were generated to identify linear relationships between features and churn.

#### Key Correlations Identified:

1. **Customer service calls**: Strong positive correlation with churn
2. **International plan**: Positive correlation with churn
3. **Total day minutes**: Moderate correlation with churn
4. **Voice mail plan**: Negative correlation with churn (retention indicator)

### Inter-Feature Correlations

- **Minutes and Charges**: Perfect correlation (charges removed as redundant)
- **Usage patterns**: Moderate correlations between day, evening, and night minutes
- **Plan features**: Low correlation between international and voice mail plans

---

## Recursive Feature Elimination (RFE)

**Method**: RFE with RandomForestClassifier (200 estimators)

**Purpose**: Identify the most predictive features for churn prediction.

**Results**:
- **Top 20 features selected** from engineered feature set
- RFE ranking provides feature importance scores
- Selected features used in final model training

**Implementation**: `src/pipelines/engineer_features.py`

---

## Customer Segmentation Analysis

### Segmentation by Usage

**Segments**:
- Low Usage: 0-500 total minutes
- Medium Usage: 500-900 total minutes
- High Usage: 900+ total minutes

**Churn Rates by Segment**:
- Low usage customers show elevated churn risk
- High usage customers also show higher churn (extreme users)
- Medium usage customers have lowest churn rates

### Segmentation by Service Calls

**Segments**:
- Low Risk: 0-1 service calls
- Medium Risk: 2-3 service calls
- High Risk: 4+ service calls

**Churn Rates by Segment**:
- Low risk: Lowest churn rate
- Medium risk: Moderate churn rate
- High risk: Highest churn rate (critical segment)

### Combined Segmentation Heatmap

A two-dimensional segmentation (Usage × Service Calls) reveals:
- **Highest risk**: High usage + High service calls
- **Lowest risk**: Medium usage + Low service calls
- Clear patterns for targeted retention campaigns

---

## Statistical Summary

### Most Significant Predictors (Ranked)

1. **Customer service calls** (p = 5.27e-18, t-test)
2. **International plan** (p = 2.49e-50, chi-square; p = 4.28e-23, t-test)
3. **Total day minutes** (p = 1.22e-20, t-test)
4. **Voice mail plan** (p = 5.15e-09, chi-square; p = 1.61e-11, t-test)
5. **Service call groups** (F = 181.65, p = 1.25e-74, ANOVA)

### Effect Sizes

- **Large effect**: International plan (Cramér's V = 0.258)
- **Moderate effect**: Customer service calls (high Cohen's d)
- **Small-moderate effect**: Voice mail plan (Cramér's V = 0.101)

---

## Key Insights

### 1. Service Calls as Primary Indicator
- Customer service calls emerged as the strongest predictor across multiple tests
- Clear dose-response relationship: more calls = higher churn risk
- Actionable insight: Early intervention for customers with 2+ service calls

### 2. International Plan Risk
- Strongest categorical association with churn
- Customers with international plans require special attention
- Potential pricing or service quality issues

### 3. Voice Mail as Retention Tool
- Negative association with churn (retention indicator)
- Customers using voice mail are more engaged
- Consider promoting voice mail features

### 4. Usage Pattern Complexity
- Both extreme low and extreme high usage correlate with churn
- Moderate, consistent usage patterns indicate stable customers
- Suggests need for usage-based segmentation

---

## Recommendations

### For Model Development
1. **Prioritize features** with highest statistical significance and effect sizes
2. **Create interaction features** combining service calls with usage patterns
3. **Use RFE results** to focus on top 20 features for model training

### For Business Action
1. **Monitor service calls**: Implement early warning system for 2+ calls
2. **Review international plan**: Investigate why international plan holders churn more
3. **Promote voice mail**: Use as retention tool for at-risk customers
4. **Target extreme users**: Develop strategies for both low and high usage segments

---

## Methodology Notes

- **Statistical significance threshold**: p < 0.05
- **Effect size interpretation**: Cohen's d for continuous, Cramér's V for categorical
- **Multiple testing**: Results interpreted with awareness of multiple comparisons
- **Assumptions**: Tests assume independence of observations (satisfied for customer data)

---

**Report Generated From**: `notebooks/02_feature_engineering.ipynb`  
**Statistical Tests**: T-tests, Chi-square tests, ANOVA, Correlation analysis  
**Visualizations**: Saved in `visualizations/interactive/` (t-test significance, chi-square effect size, correlation heatmaps)

