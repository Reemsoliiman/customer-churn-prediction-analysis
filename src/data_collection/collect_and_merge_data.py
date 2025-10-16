import pandas as pd

# Download and merge churn-bigml-20.csv and churn-bigml-80.csv from Kaggle
def collect_and_merge_data():
    # Assuming files are downloaded manually to data/raw/
    df20 = pd.read_csv('data/raw/churn-bigml-20.csv')
    df80 = pd.read_csv('data/raw/churn-bigml-80.csv')
    merged_data = pd.concat([df20, df80], axis=0, ignore_index=True)
    merged_data.to_csv('data/processed/merged_churn_data.csv', index=False)
    return merged_data

if __name__ == "__main__":
    collect_and_merge_data()