#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(
        description="Plot histogram of total claim paid; log-scale density"
    )
    p.add_argument("--input", default="severity.csv",
                   help="CSV with the five severity fields")
    p.add_argument("--out", default=None,
                   help="Filename to save plot (e.g. total_paid.png)")
    args = p.parse_args()

    # 1. Load and coerce to numeric
    df = pd.read_csv(args.input)
    for col in [
        "amountPaidOnIncreasedCostOfComplianceClaim",
        "netBuildingPaymentAmount",
        "netContentsPaymentAmount"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Compute totalPaid
    df["totalPaid"] = (
        df["netBuildingPaymentAmount"]
      + df["netContentsPaymentAmount"]
      + df["amountPaidOnIncreasedCostOfComplianceClaim"]
    ).dropna()

    # 3. Plot density‐normalized histogram
    plt.figure()
    plt.hist(df["totalPaid"], bins=50, density=True)
    plt.xscale("log")
    plt.title("Density of Total Paid (Log-Scale)")
    plt.xlabel("Total Paid (log scale)")
    plt.ylabel("Density")
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out)
        print(f"Saved histogram to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

## example usage python3 plot_severity.py --input severity.csv --out total_paid_hist.png
