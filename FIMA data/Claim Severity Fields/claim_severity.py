#!/usr/bin/env python3
import argparse
import requests
from itertools import islice

API_ENDPOINT = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
FIELDS = [
    "id",
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim",
    "netBuildingPaymentAmount",
    "netContentsPaymentAmount"
]

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def fetch_chunk(ids_chunk):
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
        description="Fetch claim‐severity fields for given FEMA claim IDs"
    )
    p.add_argument("--ids-file", required=True,
                   help="One claim ID per line")
    p.add_argument("--out", default="severity.csv",
                   help="Output CSV filename")
    p.add_argument("--chunk-size", type=int, default=25,
                   help="Number of IDs per request")
    args = p.parse_args()

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
            for line in lines:
                fout.write(line + "\n")

    print(f"Wrote {args.out} for {len(ids)} IDs.")

if __name__ == "__main__":
    main()

## example usage python3 claim_severity.py --ids-file my_ids.txt --out severity.csv
