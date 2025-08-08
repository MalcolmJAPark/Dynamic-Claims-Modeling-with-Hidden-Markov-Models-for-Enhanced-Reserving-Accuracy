# posterior_at_asofdate.py
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp, gammaln
import sys

# ---------- emissions ----------
def log_poisson_pmf(k, lam):
    lam = np.maximum(lam, 1e-8)
    return k * np.log(lam) - lam - gammaln(k + 1)

def log_lognormal_pdf(x, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    return -np.log(x * sigma * np.sqrt(2*np.pi) + 1e-16) - ((np.log(x) - mu)**2) / (2 * sigma**2)

def build_log_emissions(counts, logsev, sev_mask, lambdas, mus, sigmas):
    """
    Build T x K log-emission matrix:
      - Always includes Poisson(counts | λ_k)
      - Adds LogNormal(exp(logsev) | μ_k, σ_k) ONLY where severity is observed (sev_mask True)
    """
    T = len(counts)
    K = len(lambdas)
    log_b = np.zeros((T, K))
    # precompute observed-severity levels once
    x_obs = np.exp(logsev[sev_mask])
    for k in range(K):
        # frequency term for all t
        log_b[:, k] = log_poisson_pmf(counts, lambdas[k])
        # severity term only where observed
        sev_term_obs = log_lognormal_pdf(x_obs, mus[k], sigmas[k])
        tmp = np.zeros(T)
        tmp[sev_mask] = sev_term_obs
        log_b[:, k] += tmp
    return log_b

# ---------- filtered posterior at T ----------
def filtered_posterior_at_T(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas):
    """
    Compute P(S_T | data_{1:T}) via forward recursion (alpha) and normalizing at T.
    Returns (posterior_T, log_likelihood, T_index).
    """
    T = len(counts)
    K = len(pi)
    log_b = build_log_emissions(counts, logsev, sev_mask, lambdas, mus, sigmas)

    # forward (log-alpha)
    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(pi + 1e-16) + log_b[0]
    for t in range(1, T):
        # for each next-state j, sum over previous i
        log_alpha[t] = log_b[t] + logsumexp(log_alpha[t-1][:, None] + np.log(A + 1e-16), axis=0)

    # filtered posterior at T = normalize alpha_T
    log_alpha_T = log_alpha[-1]
    posterior_T = np.exp(log_alpha_T - logsumexp(log_alpha_T))
    ll = logsumexp(log_alpha[-1])  # sequence log-likelihood (optional)
    return posterior_T, ll, T-1

def main(agg_path="aggregated_quarterly.csv", params_path="hmm_params.json"):
    # load data
    df = pd.read_csv(agg_path, parse_dates=["quarter_start"]).sort_values("quarter_start")
    counts = df["n_claims"].astype(int).to_numpy()
    logsev = df["avg_log_severity"].to_numpy(dtype=float)
    sev_mask = ~np.isnan(logsev)

    # load fitted params
    with open(params_path, "r") as f:
        P = json.load(f)
    pi = np.asarray(P["pi"], dtype=float)
    A = np.asarray(P["A"], dtype=float)
    lambdas = np.asarray(P["lambdas"], dtype=float)
    mus = np.asarray(P["mus"], dtype=float)
    sigmas = np.asarray(P["sigmas"], dtype=float)

    # sanity checks
    if pi.shape[0] != 2 or A.shape != (2, 2):
        raise ValueError("Expected 2-state HMM params. Got pi.shape={}, A.shape={}".format(pi.shape, A.shape))
    # normalize just in case
    A = A / A.sum(axis=1, keepdims=True)
    pi = pi / pi.sum()

    # compute filtered posterior at T
    post_T, ll, idx_T = filtered_posterior_at_T(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas)
    asof = df.iloc[idx_T]["quarter_start"]

    print(f"As-of quarter (T): {asof:%Y-%m-%d}")
    print(f"Sequence log-likelihood: {ll:.4f}")
    print(f"P(S_T = low-risk | data)  = {post_T[0]:.4f}")
    print(f"P(S_T = high-risk| data)  = {post_T[1]:.4f}")

    # optional: save to JSON for downstream simulation seeding
    out = {
        "as_of_quarter": str(asof.date()),
        "posterior_T": {"low": float(post_T[0]), "high": float(post_T[1])},
        "log_likelihood": float(ll)
    }
    with open("posterior_at_T.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved posterior -> posterior_at_T.json")

if __name__ == "__main__":
    # CLI usage:
    #   python posterior_at_asofdate.py
    #   python posterior_at_asofdate.py <aggregated_quarterly.csv> <hmm_params.json>
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(agg_path=sys.argv[1])
    else:
        main(agg_path=sys.argv[1], params_path=sys.argv[2])

## example usage: python3 posterior_at_asofdate.py
