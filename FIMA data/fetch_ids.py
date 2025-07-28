#!/usr/bin/env python3
import argparse
import requests

API_ENDPOINT = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
ID_FIELD     = "id"

def fetch_ids(limit):
    params = {
        "$select": ID_FIELD,
        "$top":    limit or 100
    }
    resp = requests.get(API_ENDPOINT, params=params)
    resp.raise_for_status()
    # The API returns CSV if you ask for format=csv, but JSON by default
    data = resp.json().get("FimaNfipClaims", resp.json().get("data", []))
    return [r[ID_FIELD] for r in data]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100,
                   help="Number of IDs to fetch")
    p.add_argument("--out", default="ids.txt",
                   help="File to write IDs (one per line)")
    args = p.parse_args()

    ids = fetch_ids(args.limit)
    with open(args.out, "w") as f:
        for _id in ids:
            f.write(_id + "\n")
    print(f"Wrote {len(ids)} IDs to {args.out}")

if __name__=="__main__":
    main()
