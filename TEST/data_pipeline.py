import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def run_pipeline():
    print("--- STARTING DATA PIPELINE ---")
    
    # 1. Load Raw Data
    input_file = 'bankruptcy-prevention.xlsx'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    print(f"Loading raw data from {input_file}...")
    df_raw = pd.read_excel(input_file)
    
    # 2. Parse Semicolon Strings
    # Data is in the first column, semicolon separated
    raw_column = df_raw.iloc[:, 0]
    rows = [row.split(';') for row in raw_column]
    columns = [
        'industrial_risk', 'management_risk', 'financial_flexibility',
        'credibility', 'competitiveness', 'operating_risk', 'class'
    ]
    df = pd.DataFrame(rows, columns=columns)
    
    # 3. Clean and Format
    # Convert feature columns to numeric
    for col in columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Label Encode the class (target)
    # non-bankruptcy -> 1, bankruptcy -> 0 (based on previous mapping)
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    
    print(f"Parsed and cleaned {len(df)} rows.")
    
    # 4. Remove Duplicates (The "Organizing" step)
    df_unique = df.drop_duplicates()
    print(f"Removed {len(df) - len(df_unique)} duplicates.")
    print(f"Final dataset size: {len(df_unique)} rows.")
    
    # 5. Save Final Output
    output_file = 'bankruptcy_final.csv'
    df_unique.to_csv(output_file, index=False)
    print(f"SUCCESS: Organized data saved to {output_file}")
    
    # Print label mapping for reference
    for i, label in enumerate(le.classes_):
        print(f"Label Mapping -> {i}: {label}")

if __name__ == "__main__":
    run_pipeline()
