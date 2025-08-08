#!/usr/bin/env python3
# simulate_frequency_severity.py
#
# Seed initial states from filtered posterior at T, simulate state paths,
# then draw counts via Poisson and severities via Log-Normal (or Gamma).
# Aggregates to per-path, per-horizon totals and writes simulated_claims.csv.
#
# Usage:
#   python simulate_frequency_severity.py 5000 8 --severity lognormal --seed 2025
#
# Output:
#   - simulated_claims.csv  (path, horizon, state, n_claims, total_loss)
#   - (optional) summary percentiles printed to console

import json
import argparse
import numpy as np
import pandas as pd

def get_rng(seed=2025):
    return np.random.default_rng(seed)

def load_params(params_path="hmm_params.json", posterior_path="posterior_at_T.json"):
    with open(params_path, "r") as f:
        P = json.load(f)
    with open(posterior_path, "r") as f:
        Q = json.load(f)

    A = np.asarray(P["A"], dtype=float)
    A = A / A.sum(axis=1, keepdims=True)
    lambdas = np.asarray(P["lambdas"], dtype=float)
    mus     = np.asarray(P["mus"], dtype=float)
    sigmas  = np.asarray(P["sigmas"], dtype=float)

    pi_T = np.array([Q["posterior_T"]["low"], Q["posterior_T"]["high"]], dtype=float)
    pi_T = pi_T / pi_T.sum()
    as_of = Q.get("as_of_quarter", None)
    return pi_T, A, lambdas, mus, sigmas, as_of

def sample_initial_states(rng, pi_T, N):
    return rng.choice([0, 1], size=N, p=pi_T)

def simulate_state_paths(rng, A, initial_states, H):
    N = initial_states.shape[0]
    states = np.zeros((N, H+1), dtype=int)
    states[:, 0] = initial_states
    for h in range(1, H+1):
        prev = states[:, h-1]
        probs = A[prev]               # (N, 2)
        u = rng.random(N)
        states[:, h] = (u < probs[:, 1]).astype(int)
    return states

def simulate_counts_and_losses(rng, states, lambdas, mus, sigmas, severity="lognormal"):
    """
    Vectorized simulation of counts and aggregated losses per path/horizon.
    Returns:
      counts: (N, H+1) ints
      totals: (N, H+1) floats
    """
    N, H1 = states.shape
    counts = np.zeros((N, H1), dtype=int)
    totals = np.zeros((N, H1), dtype=float)

    # For each horizon h>=1, sample N_h ~ Poisson(lambda_{state_h})
    for h in range(1, H1):
        s_h = states[:, h]                  # (N,)
        lam_h = lambdas[s_h]                # (N,)
        n_h = rng.poisson(lam_h)            # (N,)
        counts[:, h] = n_h

        idx = np.where(n_h > 0)[0]
        if idx.size == 0:
            continue

        # expand per-claim draws
        n_claims_total = int(n_h[idx].sum())
        states_for_claims = np.repeat(s_h[idx], n_h[idx])

        if severity.lower() == "lognormal":
            sev = rng.lognormal(mean=mus[states_for_claims],
                                sigma=sigmas[states_for_claims])
        elif severity.lower() == "gamma":
            # Here we interpret mus as shape and sigmas as scale for Gamma
            sev = rng.gamma(shape=mus[states_for_claims],
                            scale=sigmas[states_for_claims])
        else:
            raise ValueError("severity must be 'lognormal' or 'gamma'.")

        # sum back to path-level
        path_index = np.repeat(idx, n_h[idx])
        totals[:, h] = np.bincount(path_index, weights=sev, minlength=N)

    return counts, totals

def main():
    ap = argparse.ArgumentParser(description="Simulate future counts & losses conditional on latent states.")
    ap.add_argument("N", type=int, nargs="?", default=5000, help="Monte Carlo paths")
    ap.add_argument("H", type=int, nargs="?", default=8, help="Future quarters to simulate")
    ap.add_argument("--severity", choices=["lognormal", "gamma"], default="lognormal")
    ap.add_argument("--params", default="hmm_params.json")
    ap.add_argument("--posterior", default="posterior_at_T.json")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--no-summary", action="store_true", help="Skip percentile summary printout")
    args = ap.parse_args()

    rng = get_rng(args.seed)
    pi_T, A, lambdas, mus, sigmas, as_of = load_params(args.params, args.posterior)

    initial = sample_initial_states(rng, pi_T, args.N)
    states  = simulate_state_paths(rng, A, initial, args.H)
    counts, totals = simulate_counts_and_losses(rng, states, lambdas, mus, sigmas, severity=args.severity)

    # Assemble per-path/horizon table (drop h=0 which is at T)
    rows = []
    for i in range(args.N):
        for h in range(1, args.H + 1):
            rows.append({
                "path": i,
                "horizon": h,
                "state": int(states[i, h]),
                "n_claims": int(counts[i, h]),
                "total_loss": float(totals[i, h])
            })
    df = pd.DataFrame.from_records(rows)
    df.to_csv("simulated_claims.csv", index=False)

    if not args.no_summary:
        pv_by_path = totals[:, 1:].sum(axis=1)   # no discount here; see simulate_ibnr_reserves.py for PV
        qs = [50, 75, 90, 95, 99]
        qv = np.quantile(pv_by_path, np.array(qs)/100.0)
        print(f"As-of quarter (T): {as_of}")
        print("Summary percentiles of total undiscounted loss over next H quarters:")
        for p, v in zip(qs, qv):
            print(f"  {p}th: {v:,.0f}")

    print("Saved simulated claims -> simulated_claims.csv")

if __name__ == "__main__":
    main()

## example usage: python3 simulate_frequency_severity.py 5000 8 --severity lognormal --seed 2025
