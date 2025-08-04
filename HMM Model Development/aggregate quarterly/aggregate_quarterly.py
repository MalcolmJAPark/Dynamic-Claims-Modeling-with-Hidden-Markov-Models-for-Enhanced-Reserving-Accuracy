# aggregate_quarterly.py

import pandas as pd
import numpy as np

def aggregate_to_quarterly(input_path: str, output_path: str):
    """
    Reads cleaned claim-level data, aggregates into a quarterly time series of:
      - n_claims: number of unique claims per quarter
      - avg_log_severity: average log severity per claim in the quarter
      - policyCount: average exposure (policy count) per quarter
    and writes the result to CSV.
    """
    # 1. Load cleaned data
    df = pd.read_csv(input_path, parse_dates=['dateOfLoss'])
    
    # 2. Compute total severity and its log
    df['total_severity'] = df['netBuildingPaymentAmount'] + df['netContentsPaymentAmount']
    # Only include rows with positive severity for log calculation
    df = df[df['total_severity'] > 0].copy()
    df['log_severity'] = np.log(df['total_severity'])
    
    # 3. Derive quarterly period from dateOfLoss
    df['quarter'] = df['dateOfLoss'].dt.to_period('Q')
    
    # 4. Aggregate
    agg = df.groupby('quarter').agg(
        n_claims=('id', 'nunique'),
        avg_log_severity=('log_severity', 'mean'),
        policyCount=('policyCount', 'mean')
    ).reset_index()
    
    # 5. Convert period to a timestamp (optional)
    agg['quarter_start'] = agg['quarter'].dt.to_timestamp()
    
    # 6. Save to CSV
    agg.to_csv(output_path, index=False)
    print(f"Aggregated {len(agg)} quarters. Output saved to '{output_path}'.")

if __name__ == "__main__":
    aggregate_to_quarterly(
        input_path="combined_data_clean.csv",
        output_path="aggregated_quarterly.csv"
    )

## example usage: python3 aggregate_quarterly.py
