# Customer Churn Data Analysis Report

**Date: October 16, 2025**

## 1. Introduction

This report presents a comprehensive analysis of customer churn data, focusing on advanced statistical analysis and feature engineering to identify key factors influencing churn. The analysis leverages a dataset of 3333 customer records with 77 features, including demographic, usage, and plan-related variables. The objectives are to explore feature relationships with churn, identify the most predictive features, and derive actionable insights to improve churn prediction models.

## 2. Statistical Analysis

To understand the relationships between features and churn, we conducted statistical tests, including t-tests and chi-squared tests, and used Recursive Feature Elimination (RFE) for feature selection. The following subsections detail the findings.

### 2.1 T-Tests for Numerical Features

Independent t-tests were performed to compare the means of numerical features between churned and non-churned customers. The results for two key features are:

- **Customer Service Calls**: The t-test yielded a t-statistic of -12.319 and a p-value of 0.000, indicating a significant difference. Churned customers have a higher average number of customer service calls, suggesting frequent interactions may be linked to dissatisfaction or issues leading to churn.
- **Total Day Minutes**: The t-test resulted in a t-statistic of -12.075 and a p-value of 0.000, showing that churned customers use significantly more day minutes on average. This could indicate higher engagement or potential cost-related dissatisfaction.

### 2.2 Chi-Squared Tests for Categorical Features

Chi-squared tests assessed the association between categorical features (e.g., state indicators, international plan, voice mail plan) and churn. Significant features (p-value < 0.05) are summarized below:

- **International plan_Yes**: Chi-squared statistic = 202.682, p-value = 5.43e-46, indicating a strong association with churn.
- **Voice mail plan_Yes**: Chi-squared statistic = 25.656, p-value = 4.08e-07, showing a significant relationship.
- **State_TX**: Chi-squared statistic = 8.387, p-value = 3.78e-03, suggesting regional influence.
- **State_NJ**: Chi-squared statistic = 6.348, p-value = 1.18e-02, indicating another regional factor.

These results highlight that having an international plan is strongly linked to churn, followed by voice mail plan and specific states like Texas and New Jersey.

### 2.3 Recursive Feature Elimination (RFE)

RFE with a logistic regression model ranked feature importance. The top 15 features are listed below:

| Feature                | Ranking |
|------------------------|---------|
| International plan_Yes | 1       |
| Customer service calls | 1       |
| Total day minutes      | 1       |
| Total eve minutes      | 1       |
| Total intl calls       | 1       |
| Total day charge       | 1       |
| Total eve charge       | 1       |
| Total intl charge      | 1       |
| Total_Minutes          | 1       |
| Total_Calls            | 1       |
| average daily usage    | 1       |
| Number vmail messages  | 2       |
| Total night minutes    | 3       |
| Total night charge     | 4       |
| Voice mail plan_Yes    | 5       |

The RFE results confirm that usage-related features (e.g., total minutes, charges) and customer service interactions are highly predictive of churn, aligning with t-test and chi-squared findings.

## 3. Conclusion

The statistical analysis identified significant predictors of churn, including customer service calls, total day minutes, international plan, and specific state indicators. These insights, combined with RFE rankings, provide a solid foundation for feature engineering and model development in subsequent milestones.