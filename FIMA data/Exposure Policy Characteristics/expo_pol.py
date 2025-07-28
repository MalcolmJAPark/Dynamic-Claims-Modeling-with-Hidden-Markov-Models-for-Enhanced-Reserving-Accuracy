#!/usr/bin/env python3
import argparse
import requests

# FEMA NFIP Claims endpoint
API_ENDPOINT = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
# Fields we want (including id so you can join back if needed)
FIELDS = [
    "id",
    "policyCount",
    "state",
    "totalBuildingInsuranceCoverage",
    "totalContentsInsuranceCoverage"
]

def download_by_ids(ids, output_path):
    """
    Streams a CSV from the FEMA API for the given IDs.
    """
    # Build OData filter: id in ('ID1','ID2',...)
    quoted = ",".join(f"'{i}'" for i in ids)
    filter_str = f"id in ({quoted})"

    params = {
        "$select": ",".join(FIELDS),
        "$format": "csv",
        "$filter": filter_str
    }

    with requests.get(API_ENDPOINT, params=params, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Wrote {output_path} for {len(ids)} IDs")

def main():
    p = argparse.ArgumentParser(
        description="Fetch exposure fields for specified FEMA NFIP Claims IDs"
    )
    p.add_argument(
        "--ids-file", required=True,
        help="Text file with one claim ID per line"
    )
    p.add_argument(
        "--out", default="exposure.csv",
        help="Output CSV file name"
    )
    args = p.parse_args()

    # Read IDs
    with open(args.ids_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        p.error("No IDs found in --ids-file")

    download_by_ids(ids, args.out)

if __name__ == "__main__":
    main()
