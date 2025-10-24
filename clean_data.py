import pandas as pd
import numpy as np
import os

print("Starting comprehensive data cleaning and verification process...")

DATA_DIR = 'data'

def clean_crsp_data(file_name, output_file_name):
    """
    Loads a CRSP data file, aggregates duplicate (permno, date) entries using
    financially-sound logic, and saves a cleaned CSV file.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    output_path = os.path.join(DATA_DIR, output_file_name)

    print(f"\n--- Processing {file_name} ---")
    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Skipping.")
        return False

    df.columns = [col.lower() for col in df.columns]

    print("Converting data types for key numeric columns...")
    numeric_cols = [
        'vol', 'divamt', 'bidlo', 'askhi', 'prc', 'openprc', 'ret', 'retx', 
        'facpr', 'facshr', 'shrout', 'cfacpr', 'cfacshr'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    full_aggregation_logic = {
        'cusip': 'last', 'ticker': 'last', 'comnam': 'last', 'nameendt': 'last',
        'shrcd': 'last', 'exchcd': 'last', 'siccd': 'last', 'ncusip': 'last',
        'shrcls': 'last', 'tsymbol': 'last', 'naics': 'last', 'primexch': 'last',
        'trdstat': 'last', 'secstat': 'last', 'permco': 'last', 'issuno': 'last',
        'hexcd': 'last', 'hsiccd': 'last',
        'dclrdt': 'last', 'paydt': 'last', 'rcrddt': 'last', 'dlstcd': 'last',
        'nextdt': 'last', 'shrenddt': 'last', 'nwperm': 'last', 'dlpdt': 'last',
        'alprcdt': 'last',
        'vol': 'sum', 'divamt': 'sum',
        'bidlo': 'min', 'askhi': 'max', 'prc': 'last', 'openprc': 'first',
        'dlprc': 'last', 'altrc': 'last',
        'facpr': 'last', 'facshr': 'last', 'cfacpr': 'last', 'cfacshr': 'last',
        'retx': 'first', 
        'vwretd': 'mean', 'vwretx': 'mean', 'ewretd': 'mean', 'ewretx': 'mean', 'sprtrn': 'mean',
        'shrout': 'last', 'mmcnt': 'sum', 'nsdinx': 'last', 'numtrd': 'sum'
    }

    aggregation_logic = {k: v for k, v in full_aggregation_logic.items() if k in df.columns}

    print("Grouping by permno and date and applying aggregation...")
    grouped = df.groupby(['permno', 'date'])
    df_agg = grouped.agg(aggregation_logic).reset_index()

    print("Recalculating total return ('ret')...")
    prc_for_calc = df_agg['prc'].replace(0, np.nan)
    dividend_yield = df_agg['divamt'] / prc_for_calc
    dividend_yield = dividend_yield.fillna(0)
    df_agg['ret'] = (1 + df_agg['retx']) * (1 + dividend_yield) - 1

    print(f"Cleaning complete. Shape of cleaned data: {df_agg.shape}")
    print(f"Saving cleaned data to {output_path}...")
    df_agg.to_csv(output_path, index=False)
    print("Save complete.")
    return True

def verify_cleaning(file_name):
    """Verifies that the cleaned file has no duplicates."""
    file_path = os.path.join(DATA_DIR, file_name)
    print(f"\n--- Verifying {file_name} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Verification failed: File not found at {file_path}")
        return

    num_duplicates = df.duplicated(subset=['permno', 'date']).sum()

    if num_duplicates == 0:
        print(f"Success: Found 0 duplicates in {file_name}.")
    else:
        print(f"Verification Failed: Found {num_duplicates} duplicates in {file_name}.")

if __name__ == '__main__':
    # if clean_crsp_data('crsp_monthly_all.csv', 'crsp_monthly_cleaned.csv'):
    #     verify_cleaning('crsp_monthly_cleaned.csv')
    
    if clean_crsp_data('crsp_daily_all.csv', 'crsp_daily_all_cleaned.csv'):
        verify_cleaning('crsp_daily_cleaned.csv')

    print("\nProcess finished.")