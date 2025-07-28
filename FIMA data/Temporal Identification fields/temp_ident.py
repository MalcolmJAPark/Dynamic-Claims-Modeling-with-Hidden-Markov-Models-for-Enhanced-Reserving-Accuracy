#!/usr/bin/env python3
import argparse
import requests

# FEMA NFIP Redacted Claims API endpoint
API_ENDPOINT = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
# Only fetch these fields
FIELDS = "id,dateOfLoss,asOfDate"

def download_csv(output_path: str, limit: int = None) -> None:
    """
    Download FEMA NFIP claims to a CSV file.
    
    :param output_path: Path to write the CSV file.
    :param limit: If provided, only fetch the first `limit` rows.
    """
    params = {
        "$select": FIELDS,
        "$format": "csv"
    }
    if limit is not None:
        params["$top"] = limit

    with requests.get(API_ENDPOINT, params=params, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    desc = f"first {limit} rows" if limit is not None else "all rows"
    print(f"Wrote {output_path} ({desc})")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch FEMA NFIP claims to CSV"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only fetch the first N rows"
    )
    parser.add_argument(
        "--out",
        default="claims.csv",
        help="Output CSV file name"
    )
    args = parser.parse_args()

    download_csv(output_path=args.out, limit=args.limit)

if __name__ == "__main__":
    main()
