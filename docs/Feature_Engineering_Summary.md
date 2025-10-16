# Feature Engineering Summary – Milestone 2

During this stage, several feature engineering steps were applied to enrich the dataset and prepare it for machine learning modeling. These steps are detailed as follows:

## 1. Column Renaming and Tenure Definition
- The column 'Account length' was renamed to 'Customer tenure' to better reflect its meaning.
- This feature represents the length of time a customer has been subscribed.

## 2. Creation of Derived Usage Features
- **Average daily usage**: Calculated as `Total_Minutes / Customer tenure`, reflecting the average number of minutes used per day by a customer.
- **Average calls per day**: Computed as `Total_Calls / Customer tenure`, representing how frequently customers make calls daily.
- **Average minutes per call**: Determined as `Total_Minutes / Total_Calls` (with safe handling for zero calls), measuring the average duration of each call.

## 3. Complaint Indicator
- A binary feature 'High service calls' was created:
  - 1 if `Customer service calls > 3`
  - 0 otherwise
- This serves as a proxy indicator of potential complaints or dissatisfaction, which may be linked to higher churn risk.

## 4. Interaction Frequency Index
- **Calls per tenure**: Calculated as `Total_Calls / Customer tenure`, capturing the intensity of customer interactions relative to their subscription length.

## 5. Feature Scaling
- Selected numerical features (`Customer tenure`, `Total_Minutes`, `average daily usage`, `Total_Calls`, `Average calls per day`, `Average minutes per call`, `Calls per tenure`) were scaled using `MinMaxScaler`.
- Scaling ensures all numerical features are brought into a common range [0,1], improving model performance and stability.

## 6. One-Hot Encoding
- Binary categorical variables (e.g., `International plan_Yes`, `Voice mail plan_Yes`, state dummies) were already encoded in Milestone 1.
- The dataset after feature engineering was stored as `df_encoded`.

## Why Feature Engineering?
- To create more informative variables that better represent customer behavior (e.g., daily usage, calls per day).
- To capture signals of dissatisfaction (e.g., high service calls) that strongly indicate churn risk.
- To ensure all features are on the same scale, preventing bias during modeling.
- To convert categorical variables into a numerical format suitable for machine learning.
- To provide a richer, cleaner, and more predictive dataset, leading to more accurate and interpretable models.

## Final Dataset
- After all feature engineering steps, the dataset included both the original cleaned features and the newly created engineered features.
- The final dataset was validated, with a shape of 3333 rows and 77 columns, ensuring completeness.
- An interim dataset with new features was saved as `data/interim/feature_engineered_data.csv`.

## Conclusion
The feature engineering process enriched the dataset with new variables that capture daily usage, call frequency, interaction intensity, and customer dissatisfaction signals. These engineered features, alongside scaling and encoding, provide a strong foundation for machine learning model development in the next milestone.