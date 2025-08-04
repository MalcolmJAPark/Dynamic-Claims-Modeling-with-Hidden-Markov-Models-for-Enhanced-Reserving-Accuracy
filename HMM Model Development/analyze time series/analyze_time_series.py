# analyze_time_series.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_time_series(input_path: str):
    """
    Loads the quarterly aggregated data, computes summary statistics,
    detects outliers, plots time series with trends and outliers,
    and plots seasonality.
    """
    # 1. Load data
    df = pd.read_csv(input_path, parse_dates=['quarter_start'])
    df = df.sort_values('quarter_start').set_index('quarter_start')
    
    # 2. Summary statistics
    mean_n = df['n_claims'].mean()
    var_n = df['n_claims'].var()
    mean_s = df['avg_log_severity'].mean()
    var_s = df['avg_log_severity'].var()
    print(f"n_claims: mean = {mean_n:.2f}, variance = {var_n:.2f}")
    print(f"avg_log_severity: mean = {mean_s:.2f}, variance = {var_s:.2f}\n")
    
    # 3. Outlier detection (|z| > 3)
    z_n = (df['n_claims'] - mean_n) / np.sqrt(var_n)
    z_s = (df['avg_log_severity'] - mean_s) / np.sqrt(var_s)
    outliers_n = df[np.abs(z_n) > 3]
    outliers_s = df[np.abs(z_s) > 3]
    print("Outlier quarters for n_claims:")
    print(outliers_n['n_claims'])
    print("\nOutlier quarters for avg_log_severity:")
    print(outliers_s['avg_log_severity'])
    
    # 4. Plot n_claims with rolling trend and outliers
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['n_claims'], label='n_claims')
    plt.plot(df.index, df['n_claims'].rolling(window=4, min_periods=1).mean(), label='4-quarter MA')
    plt.scatter(outliers_n.index, outliers_n['n_claims'], color='red', label='Outliers')
    plt.title('Quarterly Claim Counts')
    plt.xlabel('Quarter')
    plt.ylabel('n_claims')
    plt.legend()
    plt.tight_layout()
    plt.savefig('n_claims_time_series.png')
    plt.close()
    
    # 5. Plot avg_log_severity with rolling trend and outliers
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['avg_log_severity'], label='avg_log_severity')
    plt.plot(df.index, df['avg_log_severity'].rolling(window=4, min_periods=1).mean(), label='4-quarter MA')
    plt.scatter(outliers_s.index, outliers_s['avg_log_severity'], color='red', label='Outliers')
    plt.title('Quarterly Average Log Severity')
    plt.xlabel('Quarter')
    plt.ylabel('avg_log_severity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('avg_log_severity_time_series.png')
    plt.close()
    
    # 6. Seasonality: average by quarter of year
    df['season'] = df.index.quarter
    seasonal = df.groupby('season').agg(
        mean_n_claims=('n_claims', 'mean'),
        mean_avg_log_severity=('avg_log_severity', 'mean')
    )
    
    plt.figure(figsize=(8, 4))
    seasonal['mean_n_claims'].plot(kind='bar', title='Seasonality in Claim Counts')
    plt.xlabel('Quarter of Year')
    plt.ylabel('Average n_claims')
    plt.tight_layout()
    plt.savefig('n_claims_seasonality.png')
    plt.close()
    
    plt.figure(figsize=(8, 4))
    seasonal['mean_avg_log_severity'].plot(kind='bar', title='Seasonality in Avg Log Severity')
    plt.xlabel('Quarter of Year')
    plt.ylabel('Average avg_log_severity')
    plt.tight_layout()
    plt.savefig('avg_log_severity_seasonality.png')
    plt.close()

if __name__ == "__main__":
    analyze_time_series("aggregated_quarterly.csv")

## example usage: python3 analyze_time_series.py
