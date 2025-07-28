#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot monthly claim frequency per exposure"
    )
    parser.add_argument(
        "--claims",
        default="claims.csv",
        help="CSV with columns ['id','dateOfLoss','asOfDate']"
    )
    parser.add_argument(
        "--exposure",
        default="exposure.csv",
        help="CSV with columns ['id','policyCount',…]"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="If set, save plot to this file (e.g. freq.png); otherwise show it"
    )
    args = parser.parse_args()

    # 1. Load data
    claims = pd.read_csv(args.claims, parse_dates=["dateOfLoss"])
    exp   = pd.read_csv(args.exposure)

    # 2. Merge on id
    df = pd.merge(
        claims[["id","dateOfLoss"]],
        exp[["id","policyCount"]],
        on="id",
        how="inner"
    )

    # 3. Build year-month index
    df["year_month"] = df["dateOfLoss"].dt.to_period("M").dt.to_timestamp()

    # 4. Aggregate: count claims and sum exposure
    monthly = df.groupby("year_month").agg(
        claims_count = ("id", "count"),
        total_exposure = ("policyCount", "sum")
    )

    # 5. Compute frequency
    monthly["frequency"] = monthly["claims_count"] / monthly["total_exposure"]

    # 6. Plot
    plt.figure()
    plt.plot(monthly.index, monthly["frequency"], marker="o")
    plt.title("Monthly Claim Frequency per Policy")
    plt.xlabel("Month")
    plt.ylabel("Claims ÷ Policies (Frequency)")
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
