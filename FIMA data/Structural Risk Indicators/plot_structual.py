#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(
        description="Plot severity by flood zone & elevationDifference density"
    )
    p.add_argument("--input", default="structural.csv",
                   help="CSV with structural + paid‐claim fields")
    p.add_argument("--boxplot-out", default=None,
                   help="Filename to save boxplot (e.g. box.png)")
    p.add_argument("--hist-out", default=None,
                   help="Filename to save histogram (e.g. hist.png)")
    args = p.parse_args()

    # 1. Load
    df = pd.read_csv(args.input)
    # ensure numeric
    for c in ["elevationDifference",
              "amountPaidOnBuildingClaim",
              "amountPaidOnContentsClaim",
              "amountPaidOnIncreasedCostOfComplianceClaim"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2. Compute total severity
    df["severity"] = (
        df["amountPaidOnBuildingClaim"]
      + df["amountPaidOnContentsClaim"]
      + df["amountPaidOnIncreasedCostOfComplianceClaim"]
    )

    # 3. Boxplot: severity by flood zone
    plt.figure()
    df.boxplot(column="severity", by="ratedFloodZone", rot=45)
    plt.title("Claim Severity by Flood Zone")
    plt.suptitle("")  # remove default title
    plt.xlabel("Rated Flood Zone")
    plt.ylabel("Total Paid (Severity)")
    plt.tight_layout()
    if args.boxplot_out:
        plt.savefig(args.boxplot_out)
        print(f"Saved boxplot to {args.boxplot_out}")
    else:
        plt.show()

    # 4. Histogram (density) of elevationDifference
    plt.figure()
    vals = df["elevationDifference"].dropna()
    plt.hist(vals, bins=30, density=True)
    plt.title("Density of Elevation Difference")
    plt.xlabel("Elevation Difference (feet)")
    plt.ylabel("Density")
    plt.tight_layout()
    if args.hist_out:
        plt.savefig(args.hist_out)
        print(f"Saved histogram to {args.hist_out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

## example usage python3 plot_structural.py \
#  --input structural.csv \
#  --boxplot-out severity_box.png \
#  --hist-out elevation_density.png
