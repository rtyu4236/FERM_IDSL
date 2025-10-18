import pandas as pd
import os

print("Starting duplicate check for both monthly and daily data...")

DATA_DIR = 'data'

# --- Monthly Data Check ---
print("\n--- Checking Monthly Data ---")
MONTHLY_FILE_NAME = 'crsp_monthly_all.csv'
monthly_file_path = os.path.join(DATA_DIR, MONTHLY_FILE_NAME)

print(f"Loading {monthly_file_path}...")
try:
    monthly_df = pd.read_csv(monthly_file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File not found at {monthly_file_path}")
    monthly_df = None

if monthly_df is not None:
    monthly_df.columns = [col.lower() for col in monthly_df.columns]
    print("Checking for duplicates in monthly data based on 'permno' and 'date'...")
    monthly_dup_mask = monthly_df.duplicated(subset=['permno', 'date'], keep=False)
    
    if not monthly_df[monthly_dup_mask].empty:
        print(f"Found duplicates in monthly data.")
    else:
        print("No duplicate rows found in monthly data.")

# --- Daily Data Check ---
print("\n--- Checking Daily Data ---")
DAILY_FILE_NAME = 'crsp_daily_all.csv'
daily_file_path = os.path.join(DATA_DIR, DAILY_FILE_NAME)

print(f"Loading {daily_file_path}...")
try:
    daily_df = pd.read_csv(daily_file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File not found at {daily_file_path}")
    daily_df = None

if daily_df is not None:
    daily_df.columns = [col.lower() for col in daily_df.columns]
    print("Checking for duplicates in daily data based on 'permno' and 'date'...")
    daily_dup_mask = daily_df.duplicated(subset=['permno', 'date'], keep=False)
    
    if not daily_df[daily_dup_mask].empty:
        print(f"Found duplicates in daily data.")
        daily_duplicate_rows = daily_df[daily_dup_mask]
        print(f"Found {len(daily_duplicate_rows)} total rows in daily data that are part of a duplicate set.")
        print("--- Daily Duplicate Rows (showing first 5 groups) ---")
        
        daily_grouped = daily_duplicate_rows.groupby(['permno', 'date'])
        count = 0
        for (permno, date), group in daily_grouped:
            if count >= 5:
                break
            print(f"\n--- Daily Duplicate Group {count + 1}: PERMNO={permno}, DATE={date} ---")
            print(group.to_string())
            count += 1
    else:
        print("No duplicate rows found in daily data.")

print("\n\nDuplicate check finished.")