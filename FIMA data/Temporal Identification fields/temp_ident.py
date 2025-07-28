#!/usr/bin/env python3
import argparse
import requests

API_ENDPOINT = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
FIELDS       = "id,dateOfLoss,asOfDate"

def download_by_ids(ids, output_path):
    # build an OData “in” filter: id in ('ID1','ID2',…)
    # note: make sure ids list isn’t huge, or chunk it
    quoted = ",".join(f"'{i}'" for i in ids)
    filter_str = f"id in ({quoted})"

    params = {
        "$select": FIELDS,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file",   required=True,
                        help="Text file with one ID per line")
    parser.add_argument("--out",        default="claims_subset.csv",
                        help="Output CSV file name")
    args = parser.parse_args()

    # read IDs
    with open(args.ids_file) as f:
        ids = [line.strip() for line in f if line.strip()]

    download_by_ids(ids, args.out)

if __name__ == "__main__":
    main()

## example usage python3 temp_ident.py --ids-file my_ids.txt --out subset_claims.csv
