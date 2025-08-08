# aggregate_quarterly.py
import pandas as pd
import numpy as np

def aggregate_to_quarterly(input_path: str, output_path: str, fill_avg_log_sev: bool = True):
    """
    Build a clean quarterly time series for HMM fitting.

    Fixes vs. old version:
      - n_claims = nunique(id) over ALL rows (zero-paid still counts as a claim).
      - avg_log_severity = mean(log(total_severity)) over POSITIVE severities only.
      - policyCount per quarter is taken from DISTINCT values (median), not averaged over claim rows.

    Output columns:
      quarter, quarter_start, n_claims, avg_log_severity, policyCount,
      pos_claims, zero_paid_claims
    """
    df = pd.read_csv(input_path, parse_dates=["dateOfLoss"])

    # --- derive quarter keys ---
    df["quarter"] = df["dateOfLoss"].dt.to_period("Q")
    # build per-row total severity
    df["total_severity"] = (
        df["netBuildingPaymentAmount"].astype(float) +
        df["netContentsPaymentAmount"].astype(float)
    )

    # log severity only for positive amounts
    pos_mask = df["total_severity"] > 0
    df.loc[pos_mask, "log_severity"] = np.log(df.loc[pos_mask, "total_severity"])

    # --- frequency & severity aggregation ---
    freq_sev = df.groupby("quarter").agg(
        n_claims=("id", "nunique"),
        pos_claims=("total_severity", lambda s: (s > 0).sum()),
        zero_paid_claims=("total_severity", lambda s: (s == 0).sum()),
        avg_log_severity=("log_severity", "mean"),  # NaN if no positive severity that quarter
    )

    # --- exposure: do NOT average over claim rows ---
    # take the median of DISTINCT policyCount values observed in that quarter
    expo = (
        df.drop_duplicates(subset=["quarter", "policyCount"])
          .groupby("quarter")
          .agg(policyCount=("policyCount", "median"),
               n_distinct_policyCount=("policyCount", "nunique"))
    )

    # join
    agg = freq_sev.join(expo, how="left")

    # quarter start timestamp for plotting
    agg = agg.sort_index()
    agg["quarter_start"] = agg.index.to_timestamp()

    # Optional: fill avg_log_severity gaps so downstream EM doesn't choke
    filled = 0
    if fill_avg_log_sev and agg["avg_log_severity"].isna().any():
        before_na = agg["avg_log_severity"].isna().sum()
        # forward/backward interpolate on time index
        agg["avg_log_severity"] = agg["avg_log_severity"].interpolate(limit_direction="both")
        filled = before_na - agg["avg_log_severity"].isna().sum()

    # Reorder columns
    agg = agg.reset_index()[[
        "quarter", "quarter_start",
        "n_claims", "avg_log_severity", "policyCount",
        "pos_claims", "zero_paid_claims", "n_distinct_policyCount"
    ]]

    # Save
    agg.to_csv(output_path, index=False)

    # Console diagnostics
    print(f"Aggregated {len(agg)} quarters → {output_path}")
    if filled:
        print(f"Filled {filled} missing avg_log_severity values via interpolation.")
    # warn if exposure varies within a quarter (likely a data issue)
    bad = agg.loc[agg["n_distinct_policyCount"] > 1, "quarter"].tolist()
    if bad:
        print("WARNING: Multiple distinct policyCount values observed in quarters:", bad)
        print("         Using median per quarter. Consider sourcing exposure from a dedicated table.")

if __name__ == "__main__":
    aggregate_to_quarterly("combined_data_clean.csv", "aggregated_quarterly.csv", fill_avg_log_sev=True)


## example usage: python3 aggregate_quarterly.py
