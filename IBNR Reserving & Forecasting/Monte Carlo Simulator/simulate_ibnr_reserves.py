# simulate_ibnr_reserves.py
# -------------------------------------------------
# This script fixes semantics and naming:
# - By default it computes **Ultimate over next H quarters** (future occurrences),
#   NOT strict IBNR at T (that's in strict_ibnr_at_T.py).
# - It optionally discounts cashflows by (h + reporting_lag) if you pass a delay distribution.
# - It reads fitted HMM params from hmm_params.json and P(S_T | data) from posterior_at_T.json,
#   seeds the initial state distribution with that posterior, simulates latent states,
#   then simulates counts & severities conditional on state.
#
# CLI:
#   python simulate_ibnr_reserves.py [N] [H] [r] [--discount-by-reporting-lag]
#
# Example:
#   python simulate_ibnr_reserves.py 5000 8 0.0 --discount-by-reporting-lag
#
# Output:
#   - simulated_claims.csv (per-path, per-horizon totals)
#   - simulated_reserves_summary.json (percentiles)
#
import json
import argparse
import numpy as np
import pandas as pd

# ---------------- RNG ----------------
def get_rng(seed=2025):
    return np.random.default_rng(seed)

# ---------------- Inputs ----------------
def load_fitted_params(params_path="hmm_params.json", posterior_path="posterior_at_T.json"):
    with open(params_path, "r") as f:
        P = json.load(f)
    with open(posterior_path, "r") as f:
        Q = json.load(f)

    A = np.asarray(P["A"], dtype=float)
    A = A / A.sum(axis=1, keepdims=True)  # row-normalize
    lambdas = np.asarray(P["lambdas"], dtype=float)
    mus = np.asarray(P["mus"], dtype=float)
    sigmas = np.asarray(P["sigmas"], dtype=float)

    # Filtered posterior at T
    pi_T = np.array([Q["posterior_T"]["low"], Q["posterior_T"]["high"]], dtype=float)
    pi_T = pi_T / pi_T.sum()

    return pi_T, A, lambdas, mus, sigmas, Q.get("as_of_quarter", None)

# ---------------- State simulation ----------------
def sample_initial_states(rng, pi_T, N):
    return rng.choice([0, 1], size=N, p=pi_T)

def simulate_state_paths(rng, A, initial_states, H):
    N = initial_states.shape[0]
    states = np.zeros((N, H+1), dtype=int)
    states[:, 0] = initial_states
    for h in range(1, H+1):
        prev = states[:, h-1]
        probs = A[prev]                      # (N, 2)
        u = rng.random(N)
        states[:, h] = (u < probs[:, 1]).astype(int)
    return states

# ---------------- Frequency & Severity ----------------
def simulate_counts_and_losses(rng, states, lambdas, mus, sigmas,
                               severity_dist="lognormal"):
    """
    Returns:
      counts: (N, H+1) int, counts at each horizon (col 0 is initial state at T -> zero by construction)
      totals: (N, H+1) float, total loss per path/horizon
    """
    N, H1 = states.shape
    counts = np.zeros((N, H1), dtype=int)
    totals = np.zeros((N, H1), dtype=float)

    for h in range(1, H1):
        s_h = states[:, h]
        lam_h = lambdas[s_h]
        n_h = rng.poisson(lam_h)            # vectorized Poisson by path
        counts[:, h] = n_h

        ix_pos = np.where(n_h > 0)[0]
        if ix_pos.size == 0:
            continue
        s_pos = s_h[ix_pos]
        n_pos = n_h[ix_pos]

        # Draw severities claim-by-claim; do it in chunks
        # Build a flat array of per-claim state params, then sum per path
        per_claim_sizes = n_pos
        total_claims = int(per_claim_sizes.sum())
        if total_claims == 0:
            continue

        state_for_claims = np.repeat(s_pos, per_claim_sizes)
        if severity_dist == "lognormal":
            sev = rng.lognormal(mean=mus[state_for_claims],
                                sigma=sigmas[state_for_claims])
        elif severity_dist == "gamma":
            # Interpret mus as shape and sigmas as scale when gamma chosen
            sev = rng.gamma(shape=mus[state_for_claims],
                            scale=sigmas[state_for_claims])
        else:
            raise ValueError("severity_dist must be 'lognormal' or 'gamma'.")

        # Sum back per path
        # build an index per-claim -> path
        path_index = np.repeat(ix_pos, per_claim_sizes)
        totals[:, h] = np.bincount(path_index, weights=sev, minlength=N)

    return counts, totals

# ---------------- Reporting lags (optional) ----------------
def draw_reporting_lags(rng, n_claims_vec, delay_ds, delay_ps):
    """
    For each path's count at a horizon, draw that many lags from discrete {delay_ds, delay_ps}.
    Returns a list of 1D arrays of lags per path for that horizon.
    """
    out = []
    for n in n_claims_vec:
        if n <= 0:
            out.append(np.empty(0, dtype=int))
        else:
            out.append(rng.choice(delay_ds, size=n, p=delay_ps))
    return out

# ---------------- Reserve assembly ----------------
def compute_reserves(counts, totals, r=0.0, discount_by_reporting_lag=False,
                     rng=None, delay_ds=None, delay_ps=None):
    """
    Ultimate next H (NOT strict IBNR at T):
      - reserve_raw = sum of totals over horizons 1..H
      - reserve_pv  = discounted by (1+r)^-h, or by (1+r)^-(h + L) if discount_by_reporting_lag=True

    If discount_by_reporting_lag=True, delay_ds and delay_ps must be provided.
    """
    if rng is None:
        rng = get_rng()

    N, H1 = counts.shape
    H = H1 - 1

    # Raw (no discount)
    reserve_raw = totals[:, 1:].sum(axis=1)

    # PV with per-horizon discount
    if not discount_by_reporting_lag:
        disc = (1.0 + r) ** -np.arange(1, H+1)      # shape (H,)
        reserve_pv = (totals[:, 1:] * disc).sum(axis=1)
    else:
        if delay_ds is None or delay_ps is None:
            raise ValueError("Pass delay_ds and delay_ps when discount_by_reporting_lag=True")
        delay_ds = np.asarray(delay_ds, dtype=int)
        delay_ps = np.asarray(delay_ps, dtype=float)
        delay_ps = delay_ps / delay_ps.sum()

        reserve_pv = np.zeros(N)
        # For each horizon, discount each claim by h + L (reporting lag)
        for h in range(1, H+1):
            n_vec = counts[:, h]
            lags_per_path = draw_reporting_lags(rng, n_vec, delay_ds, delay_ps)
            # Need per-claim loss amounts to discount by different (h+L)
            # Approximate by splitting each path's total equally across its claims if present.
            # (If you want exact per-claim assignment, return per-claim severities from the simulator.)
            for i in range(N):
                n_i = n_vec[i]
                if n_i <= 0:
                    continue
                # Flat split approximation
                avg_loss = totals[i, h] / n_i
                Ls = lags_per_path[i]
                pv = np.sum(avg_loss * (1.0 + r) ** (-(h + Ls)))
                reserve_pv[i] += pv

    return reserve_raw, reserve_pv

# ---------------- Main driver ----------------
def main():
    parser = argparse.ArgumentParser(description="Simulate ULTIMATE next-H reserves from HMM states.")
    parser.add_argument("N", type=int, nargs="?", default=5000, help="Monte Carlo paths")
    parser.add_argument("H", type=int, nargs="?", default=8, help="Projection horizon (quarters)")
    parser.add_argument("r", type=float, nargs="?", default=0.0, help="Per-quarter discount rate")
    parser.add_argument("--severity", choices=["lognormal", "gamma"], default="lognormal")
    parser.add_argument("--discount-by-reporting-lag", action="store_true",
                        help="Discount each claim by (h + lag). Requires delay distribution.")
    parser.add_argument("--delay-ds", type=int, nargs="*", default=[0,1,2,3,4],
                        help="Support for reporting delay (quarters)")
    parser.add_argument("--delay-ps", type=float, nargs="*", default=[0.50,0.30,0.10,0.07,0.03],
                        help="Probabilities for reporting delay; will be renormalized.")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    rng = get_rng(args.seed)

    # 1) Load fitted params + posterior at T
    pi_T, A, lambdas, mus, sigmas, as_of = load_fitted_params()

    # 2) Simulate states -> counts & losses
    initial_states = sample_initial_states(rng, pi_T, args.N)
    states = simulate_state_paths(rng, A, initial_states, args.H)
    counts, totals = simulate_counts_and_losses(rng, states, lambdas, mus, sigmas, severity_dist=args.severity)

    # 3) Compute reserves (Ultimate next H)
    if args.discount_by_reporting_lag:
        reserve_raw, reserve_pv = compute_reserves(
            counts, totals, r=args.r, discount_by_reporting_lag=True,
            rng=rng, delay_ds=args.delay_ds, delay_ps=args.delay_ps
        )
    else:
        reserve_raw, reserve_pv = compute_reserves(counts, totals, r=args.r, discount_by_reporting_lag=False)

    # 4) Persist per-path by horizon and summaries
    # Per-path/horizon table
    records = []
    N, H1 = counts.shape
    for i in range(N):
        for h in range(1, H1):
            records.append({
                "path": i,
                "horizon": h,
                "state": states[i, h],
                "n_claims": int(counts[i, h]),
                "total_loss": float(totals[i, h])
            })
    df = pd.DataFrame.from_records(records)
    df.to_csv("simulated_claims.csv", index=False)

    # Summary percentiles on PV reserves
    percentiles = [50, 75, 90, 95, 99]
    q = np.quantile(reserve_pv, np.array(percentiles)/100.0)
    summary = {
        "as_of_quarter": as_of,
        "N": int(args.N),
        "H": int(args.H),
        "discount_rate_per_quarter": args.r,
        "severity_dist": args.severity,
        "discount_by_reporting_lag": bool(args.discount_by_reporting_lag),
        "reserve_pv_percentiles": {str(p): float(v) for p, v in zip(percentiles, q)}
    }
    with open("simulated_reserves_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved per-path totals -> simulated_claims.csv")
    print("Saved PV reserve percentiles -> simulated_reserves_summary.json")
    print("Example percentiles (PV):")
    for p, v in zip(percentiles, q):
        print(f"  {p}th: {v:,.0f}")

if __name__ == "__main__":
    main()

## example usage: python3 simulate_ibnr_reserves.py 5000 8 0.0 --discount-by-reporting-lag
