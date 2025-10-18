import yaml

from src.data_collection.collect_and_merge_data import collect_and_merge
from src.data_collection.validate_dataset import verify_data_files

# from src.preprocessing.handle_missing_values import impute_missing
# from src.preprocessing.outlier_detection import remove_outliers
# from src.preprocessing.feature_scaling import scale_features
# from src.preprocessing.categorical_encoding import encode_categorical

# from src.modeling.data_splitter import split_data
# from src.modeling.logistic_regression_model import train_logistic_regression
# from src.modeling.model_evaluator import evaluate_model

# from src.deployment.fastapi_churn_predictor import deploy_model

# from src.monitoring.model_performance_tracker import track_performance

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()

    # Step 1: Verify Data Files
    verified_files = verify_data_files(
        config['data']['raw']['churn_bigml_20'],
        config['data']['raw']['churn_bigml_80']
    )

    # Step 2: Data Collection and Merging
    merged_data_path = collect_and_merge(
        config['data']['raw']['churn_bigml_20'],
        config['data']['raw']['churn_bigml_80'],
        output_path=config['data']['processed']['merged_churn_data']
    )

    # # Step 3: Preprocessing
    # imputed_data = impute_missing(merged_data_path, config['preprocessing']['imputation_method'])
    # cleaned_data = remove_outliers(imputed_data, config['preprocessing']['outlier_threshold'])
    # scaled_data = scale_features(cleaned_data, config['preprocessing']['scaler'])
    # encoded_data = encode_categorical(scaled_data, config['preprocessing']['encoding'])

    # # Step 4: Modeling
    # X_train, X_test, y_train, y_test = split_data(encoded_data, config['modeling']['test_size'])
    # model = train_logistic_regression(X_train, y_train, config['modeling']['logistic_params'])
    # metrics = evaluate_model(model, X_test, y_test)

    # # Step 5: Deployment
    # deploy_model(model, config['deployment']['api_endpoint'])

    # # Step 6: Monitoring
    # track_performance(metrics, config['monitoring']['log_path'])

if __name__ == "__main__":
    main()