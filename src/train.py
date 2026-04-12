import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score # Accuracy use kar rahe hain classification ke liye
from sklearn.model_selection import train_test_split
import argparse

def train(bucket_name):
    # 1. Path Update: Aapka folder 'DATA' (capital) hai aur file ka naam badal gaya hai
    data_path = f"gs://{bucket_name}/DATA/loan_approval_dataset.csv"
    df = pd.read_csv(data_path)
    
    # 2. Dataset Cleaning: Column names mein extra spaces ho sakte hain
    df.columns = df.columns.str.strip()
    
    # 3. Features & Target: Loan dataset ke hisaab se (Example logic)
    # Maan lijiye 'loan_status' target hai aur baaki features
    X = df.drop(columns=['loan_id', 'loan_status']) 
    y = df['loan_status'].apply(lambda x: 1 if x.strip() == 'Approved' else 0)
    
    # Categorical data ko handle karne ke liye (Simple Encoding)
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    with mlflow.start_run():
        # Classification model use karenge loan approval ke liye
        model = xgb.XGBClassifier() 
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        # Save model
        model.save_model("model.bst")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str)
    args = parser.parse_args()
    train(args.bucket_name)