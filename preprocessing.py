# Fill missing values: numeric columns with median, categorical columns with mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Identify categorical columns (excluding target 'Churn' if present)
categorical_cols = df.select_dtypes(include=['object']).columns.drop('Churn', errors='ignore')

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df_encoded.head()
