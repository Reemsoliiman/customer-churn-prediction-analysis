import pandas as pd

def collect_and_merge(file_20_path, file_80_path, output_path):
    """
    Merge churn-bigml-20.csv and churn-bigml-80.csv into a single dataset.
    
    Args:
        file_20_path (str): Path to churn-bigml-20.csv
        file_80_path (str): Path to churn-bigml-80.csv
        output_path (str): Path to save merged dataset
    
    Returns:
        str: Path to the merged dataset
    """
    df20 = pd.read_csv(file_20_path)
    df80 = pd.read_csv(file_80_path)
    merged_data = pd.concat([df20, df80], axis=0, ignore_index=True)
    merged_data.to_csv(output_path, index=False)
    return output_path

# if __name__ == "__main__":
#     collect_and_merge(
#         'data/raw/churn-bigml-20.csv',
#         'data/raw/churn-bigml-80.csv',
#         'data/processed/merged_churn_data.csv'
#     )
#     print("Data collection and merging completed.")