#!/usr/bin/env python3
# initial_parameters.py

import pandas as pd
import numpy as np

def main():
    # 1) Load & sort
    df = pd.read_csv('combined_data.csv', parse_dates=['dateOfLoss'])
    df.sort_values('dateOfLoss', inplace=True)

    # 2) Extract your count & severity series
    #    – here we use 'policyCount' as the frequency
    #    – and total paid = building + contents as the severity
    counts   = df['policyCount'].values
    severity = (df['netBuildingPaymentAmount'] + df['netContentsPaymentAmount']).values

    # 3) Clean up any non‑positive severities (log‑normal only defined for >0)
    severity = np.clip(severity, 1e-6, None)

    # 4) MOM estimates
    lambda_hat = counts.mean()
    log_sev   = np.log(severity)
    mu_hat     = log_sev.mean()
    sigma_hat  = log_sev.std(ddof=0)  # population std

    # 5) Report
    print(f"Initial estimates:")
    print(f"  λ̂  = {lambda_hat:.4f} (mean count)")
    print(f"  μ̂  = {mu_hat:.4f} (mean log-severity)")
    print(f"  σ̂  = {sigma_hat:.4f} (std  log-severity)")

if __name__ == '__main__':
    main()


## example usages python3 initial_parameters.py 
# Initial estimates:
#  λ̂  = 1.7200 (mean count)
#  μ̂  = 3.7442 (mean log-severity)
#  σ̂  = 10.2208 (std  log-severity)
