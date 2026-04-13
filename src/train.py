import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import os
from google.cloud import storage

def train(bucket_name):
    # 1. Data Loading
    data_path = f"gs://{bucket_name}/DATA/loan_approval_dataset.csv"
    df = pd.read_csv(data_path)
    
    # 2. Cleaning
    df.columns = df.columns.str.strip()
    X = df.drop(columns=['loan_id', 'loan_status']) 
    y = df['loan_status'].apply(lambda x: 1 if x.strip() == 'Approved' else 0)
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    with mlflow.start_run():
        model = xgb.XGBClassifier() 
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", acc)
        
        # Vertex AI Registration Logic
        model_dir = os.environ.get('AIP_MODEL_DIR')
        if model_dir:
            local_path = "model.bst"
            model.save_model(local_path)
            
            # GCS Upload
            client = storage.Client()
            parts = model_dir.replace("gs://", "").split("/")
            bucket_target = parts[0]
            blob_path = "/".join(parts[1:]) + "model.bst"
            
            bucket = client.bucket(bucket_target)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Model sent to Registry via: {model_dir}")
        else:
            model.save_model("model.bst")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str)
    args = parser.parse_args()
    train(args.bucket_name)