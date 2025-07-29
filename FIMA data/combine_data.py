#!/usr/bin/env python3
"""
combine_data.py

Reads `my_ids.txt` (one UUID per line) and the four CSVs
severity.csv, structural.csv, exposure.csv, subset_claims.csv,
filters each to only those IDs, then merges them all into
combined_data.csv.
"""

import pandas as pd

def load_and_filter(csv_path, ids):
    df = pd.read_csv(csv_path)
    return df[df['id'].isin(ids)]

def main():
    # 1) Read the list of IDs
    with open('my_ids.txt', 'r') as f:
        my_ids = [line.strip() for line in f if line.strip()]

    # 2) Load & filter each dataset
    df_dates    = load_and_filter('subset_claims.csv',            my_ids)
    df_exposure = load_and_filter('exposure.csv',                 my_ids)
    df_struct   = load_and_filter('structural.csv',               my_ids)
    df_severity = load_and_filter('severity.csv',                 my_ids)

    # 3) Start with a DataFrame of all IDs (preserves original order)
    combined = pd.DataFrame({'id': my_ids})

    # 4) Merge in turn (left‑joins preserve every ID, even if missing data)
    combined = combined.merge(df_dates,    on='id', how='left')
    combined = combined.merge(df_exposure, on='id', how='left')
    combined = combined.merge(df_struct,   on='id', how='left')
    combined = combined.merge(df_severity, on='id', how='left')

    # 5) Write out
    combined.to_csv('combined_data.csv', index=False)
    print("✅ Wrote merged data to combined_data.csv")

if __name__ == '__main__':
    main()
