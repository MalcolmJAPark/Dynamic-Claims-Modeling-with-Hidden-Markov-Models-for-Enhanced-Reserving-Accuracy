#!/usr/bin/env python3
import argparse
import requests
from itertools import islice

API_ENDPOINT = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
FIELDS       = ["id","dateOfLoss","asOfDate"]

def chunked(it, size):
    it = iter(it)
    while True:
        c = list(islice(it, size))
        if not c: break
        yield c

def fetch_chunk(ids_chunk):
    # build "id eq '…' or id eq '…'" filter
    clauses = [f"id eq '{i}'" for i in ids_chunk]
    flt = " or ".join(clauses)
    params = {
        "$select": ",".join(FIELDS),
        "$filter": flt,
        "$format": "csv"
    }
    r = requests.get(API_ENDPOINT, params=params, stream=True)
    r.raise_for_status()
    return r.iter_lines(decode_unicode=True)

def main():
    p = argparse.ArgumentParser(
        description="Fetch id, dateOfLoss, asOfDate for a list of FEMA claim IDs"
    )
    p.add_argument("--ids-file", required=True,
                   help="One claim ID per line")
    p.add_argument("--out", default="claims.csv",
                   help="Output CSV filename")
    p.add_argument("--chunk-size", type=int, default=25,
                   help="How many IDs per request")
    args = p.parse_args()

    # read all IDs
    with open(args.ids_file) as f:
        ids = [l.strip() for l in f if l.strip()]
    if not ids:
        p.error("No IDs in " + args.ids_file)

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

    print(f"Wrote {args.out} ({len(ids)} IDs in {args.chunk_size}-size chunks)")

if __name__=="__main__":
    main()

## example usage python3 temp_ident.py --ids-file ids.txt --out claims.csv
