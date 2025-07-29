#!/usr/bin/env python3
import argparse
import requests
from itertools import islice

API_ENDPOINT = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
FIELDS = [
    "id",
    "ratedFloodZone",
    "elevationDifference",
    "postFIRMConstructionIndicator",
    "primaryResidenceIndicator",
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim"
]

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def fetch_chunk(ids_chunk):
    # build filter: id eq '…' or id eq '…'
    clauses = [f"id eq '{i}'" for i in ids_chunk]
    filter_str = " or ".join(clauses)
    params = {
        "$select": ",".join(FIELDS),
        "$filter": filter_str,
        "$format": "csv"
    }
    r = requests.get(API_ENDPOINT, params=params, stream=True)
    r.raise_for_status()
    return r.iter_lines(decode_unicode=True)

def main():
    p = argparse.ArgumentParser(
        description="Fetch structural/risk fields + paid amounts for given FEMA claim IDs"
    )
    p.add_argument("--ids-file", required=True,
                   help="One claim ID per line")
    p.add_argument("--out", default="structural.csv",
                   help="Output CSV filename")
    p.add_argument("--chunk-size", type=int, default=25,
                   help="Number of IDs per API request")
    args = p.parse_args()

    # load IDs
    with open(args.ids_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        p.error("No IDs found in --ids-file")

    first = True
    with open(args.out, "w", newline="") as fout:
        for chunk in chunked(ids, args.chunk_size):
            lines = fetch_chunk(chunk)
            header = next(lines)
            if first:
                fout.write(header + "\n")
                first = False
            # skip header on subsequent chunks
            for line in lines:
                fout.write(line + "\n")

    print(f"Wrote {args.out} for {len(ids)} IDs.")

if __name__ == "__main__":
    main()

## example usage python3 struct_risk.py --ids-file ids.txt --out structural.csv

