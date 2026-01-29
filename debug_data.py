import pandas as pd

try:
    df = pd.read_csv("dataset.csv")
    print("Columns:", df.columns.tolist())
    print("\nFirst row types:")
    print(df.dtypes)
    print("\nLast 3 columns:")
    print(df.iloc[:, -3:].head())
    
    # Check for object columns in the first n-1 columns
    X = df.iloc[:, :-1]
    obj_cols = X.select_dtypes(include=['object']).columns
    print(f"\nNon-numeric feature columns: {obj_cols.tolist()}")
except Exception as e:
    print(e)
