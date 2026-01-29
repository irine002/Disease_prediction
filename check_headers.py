import pandas as pd

files = ["symptom_Description.csv", "symptom_precaution.csv"]

for f in files:
    print(f"--- {f} ---")
    try:
        df = pd.read_csv(f)
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(2))
    except Exception as e:
        print(f"Error reading {f}: {e}")
    print("\n")
