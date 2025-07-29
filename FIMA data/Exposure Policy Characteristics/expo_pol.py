#!/usr/bin/env python3
import argparse
import requests
from itertools import islice

API_ENDPOINT = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
FIELDS = [
    "id",
    "policyCount",
    "state",
    "totalBuildingInsuranceCoverage",
    "totalContentsInsuranceCoverage"
]

def chunked(iterable, size):
    """Yield successive size-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def fetch_chunk(ids_chunk):
    """Fetch a CSV lines iterator for one chunk of IDs."""
    # build "id eq '…' or id eq '…'" filter
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
    parser = argparse.ArgumentParser(
        description="Fetch exposure fields for a list of FEMA claim IDs"
    )
    parser.add_argument(
        "--ids-file", required=True,
        help="Text file with one claim ID per line"
    )
    parser.add_argument(
        "--out", default="exposure.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=25,
        help="How many IDs to include per request"
    )
    args = parser.parse_args()

    # load IDs
    with open(args.ids_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        parser.error("No IDs found in --ids-file")

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

    print(f"Wrote {args.out} for {len(ids)} IDs using {args.chunk_size}-ID chunks.")

if __name__ == "__main__":
    main()

## example usage python3 expo_pol.py --ids-file ids.txt --out exposure.csv

