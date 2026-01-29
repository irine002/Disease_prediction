import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train():
    print("Loading dataset...")
    try:
        df = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        print("Error: dataset.csv not found. Please run download_data.py first.")
        return

    # Dimensions
    print(f"Dataset shape: {df.shape}")
    
    # Last column is probability/prognosis, but looking at the CSV snippet, 
    # there is a 'prognosis' column at the end. 
    # The snippet showed: ... ,prognosis,
    # and then values like 1,0,...,Fungal infection,
    # It seems the last column is indeed the label.
    
    # Cleaning: Drop columns that are completely empty (like trailing comma artifact)
    df.dropna(axis=1, how='all', inplace=True)
    
    # Check if 'prognosis' exists
    target_col = 'prognosis'
    if target_col not in df.columns:
        # Fallback: assume last column is target if name doesn't match
        target_col = df.columns[-1]
    
    print(f"Target column: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    
    # Save model and feature names
    print("Saving model and feature data...")
    joblib.dump(rf_model, "disease_model.pkl")
    joblib.dump(X.columns.tolist(), "symptom_columns.pkl")
    print("Done!")

if __name__ == "__main__":
    train()
