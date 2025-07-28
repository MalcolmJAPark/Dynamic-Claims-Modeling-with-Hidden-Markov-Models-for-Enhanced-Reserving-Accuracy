#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot monthly claims frequency from a FEMA NFIP CSV"
    )
    parser.add_argument(
        "-i", "--input",
        default="claims.csv",
        help="Path to the CSV file (must include a 'dateOfLoss' column)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="If set, save plot to this file (e.g. frequency.png); otherwise show on screen"
    )
    args = parser.parse_args()

    # 1. Load and parse dates
    df = pd.read_csv(args.input, parse_dates=["dateOfLoss"])
    df = df.dropna(subset=["dateOfLoss"])

    # 2. Aggregate counts by year-month
    df["year_month"] = df["dateOfLoss"].dt.to_period("M").dt.to_timestamp()
    monthly_counts = df.groupby("year_month").size()

    # 3. Plot
    plt.figure()
    plt.plot(monthly_counts.index, monthly_counts.values)
    plt.title("Monthly Claims Frequency")
    plt.xlabel("Month")
    plt.ylabel("Number of Claims")
    plt.tight_layout()

    # 4. Output
    if args.output:
        plt.savefig(args.output)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
