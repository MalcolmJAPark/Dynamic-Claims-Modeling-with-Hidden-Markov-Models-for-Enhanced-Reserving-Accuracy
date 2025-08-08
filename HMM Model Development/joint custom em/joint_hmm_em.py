# joint_hmm_em.py
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp, gammaln

# -------------------- utils --------------------
def log_poisson_pmf(k, lam):
    lam = np.maximum(lam, 1e-8)
    return k * np.log(lam) - lam - gammaln(k + 1)

def log_lognormal_pdf(x, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    return -np.log(x * sigma * np.sqrt(2*np.pi) + 1e-16) - ((np.log(x) - mu)**2) / (2 * sigma**2)

def build_log_emissions(counts, logsev, sev_mask, lambdas, mus, sigmas):
    """
    Return T x K matrix of log emission probs.
    Frequency (Poisson) always included.
    Severity term (LogNormal on exp(logsev)) only added where sev_mask==True.
    """
    T = len(counts)
    K = len(lambdas)
    log_b = np.zeros((T, K))
    # Precompute x only for observed severities
    x_obs = np.exp(logsev[sev_mask])
    for k in range(K):
        # frequency term
        log_b[:, k] = log_poisson_pmf(counts, lambdas[k])
        # severity term only where observed
        sev_term_obs = log_lognormal_pdf(x_obs, mus[k], sigmas[k])
        # insert into the right rows, leave 0 elsewhere
        tmp = np.zeros(T)
        tmp[sev_mask] = sev_term_obs
        log_b[:, k] += tmp
    return log_b

# -------------------- EM steps --------------------
def e_step(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas):
    T, K = len(counts), len(pi)
    log_b = build_log_emissions(counts, logsev, sev_mask, lambdas, mus, sigmas)

    # forward
    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(pi + 1e-16) + log_b[0]
    for t in range(1, T):
        prev = log_alpha[t-1] + np.log(A + 1e-16).T  # K x K (from-state in rows)
        log_alpha[t] = log_b[t] + logsumexp(prev, axis=1)

    # backward
    log_beta = np.zeros((T, K))
    for t in range(T-2, -1, -1):
        nxt = np.log(A + 1e-16) + (log_b[t+1] + log_beta[t+1])[None, :]  # K x K
        log_beta[t] = logsumexp(nxt, axis=1)

    # posteriors
    log_gamma = log_alpha + log_beta
    gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))

    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        m = (log_alpha[t][:, None]
             + np.log(A + 1e-16)
             + log_b[t+1][None, :]
             + log_beta[t+1][None, :])
        xi[t] = np.exp(m - logsumexp(m))
    ll = logsumexp(log_alpha[-1])
    return gamma, xi, ll

def m_step(counts, logsev, sev_mask, gamma, xi):
    T, K = gamma.shape

    # initial distribution
    pi_new = gamma[0].copy()

    # transitions
    A_new = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-16)
    A_new = A_new / A_new.sum(axis=1, keepdims=True)  # row-normalize

    # frequency params (Poisson means)
    lambdas_new = (gamma * counts[:, None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-16)
    lambdas_new = np.maximum(lambdas_new, 1e-8)

    # severity params (LogNormal on claim-level, but we use avg_log_severity proxy)
    # Use only rows where severity is observed
    g_obs = gamma[sev_mask]  # (#obs) x K
    y_obs = logsev[sev_mask]
    denom = g_obs.sum(axis=0) + 1e-16
    mus_new = (g_obs * y_obs[:, None]).sum(axis=0) / denom
    sigmas_new = np.sqrt(((g_obs * (y_obs[:, None] - mus_new)**2).sum(axis=0)) / denom)
    sigmas_new = np.maximum(sigmas_new, 1e-6)

    return pi_new, A_new, lambdas_new, mus_new, sigmas_new

def fit_em(counts, logsev, sev_mask, max_iter=200, tol=1e-4, seed=42):
    np.random.seed(seed)
    # init (2 states)
    K = 2
    mean_c = counts.mean()
    obs = logsev[sev_mask]
    mls, sls = (obs.mean() if obs.size else 0.0), (obs.std() if obs.size else 1.0)

    pi = np.array([0.5, 0.5])
    A  = np.array([[0.9, 0.1],
                   [0.1, 0.9]])
    lambdas = np.array([0.5*mean_c, 1.5*mean_c]).clip(1e-8, None)
    mus     = np.array([mls - 0.3*sls, mls + 0.3*sls])
    sigmas  = np.array([sls, sls]).clip(1e-6, None)

    ll_hist = []
    for it in range(max_iter):
        gamma, xi, ll = e_step(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas)
        ll_hist.append(ll)
        if len(ll_hist) >= 2 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            print(f"EM converged at iter {it}, ll={ll:.4f}")
            break
        pi, A, lambdas, mus, sigmas = m_step(counts, logsev, sev_mask, gamma, xi)

    return (pi, A, lambdas, mus, sigmas), ll_hist

# -------------------- Viterbi --------------------
def viterbi(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas):
    T, K = len(counts), len(pi)
    log_b = build_log_emissions(counts, logsev, sev_mask, lambdas, mus, sigmas)

    delta = np.zeros((T, K))
    psi   = np.zeros((T, K), dtype=int)
    delta[0] = np.log(pi + 1e-16) + log_b[0]
    for t in range(1, T):
        for k in range(K):
            prev = delta[t-1] + np.log(A[:, k] + 1e-16)
            psi[t, k] = np.argmax(prev)
            delta[t, k] = prev[psi[t, k]] + log_b[t, k]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]
    return states

# -------------------- main --------------------
if __name__ == "__main__":
    df = pd.read_csv("aggregated_quarterly.csv", parse_dates=["quarter_start"]).sort_values("quarter_start")
    counts = df["n_claims"].astype(int).to_numpy()
    logsev = df["avg_log_severity"].to_numpy(dtype=float)
    sev_mask = ~np.isnan(logsev)

    (pi, A, lambdas, mus, sigmas), ll_hist = fit_em(counts, logsev, sev_mask, max_iter=200, tol=1e-4)

    print("\nFitted parameters:")
    print("pi:", np.round(pi, 6).tolist())
    print("A:\n", np.round(A, 6))
    for k in range(2):
        print(f"State {k}: lambda={lambdas[k]:.4f}, mu={mus[k]:.4f}, sigma={sigmas[k]:.4f}")

    # decode
    states = viterbi(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas)
    df_out = pd.DataFrame({
        "quarter_start": df["quarter_start"].dt.strftime("%Y-%m-%d"),
        "n_claims": counts,
        "avg_log_severity": logsev,
        "sev_observed": sev_mask.astype(int),
        "state": states
    })
    df_out.to_csv("decoded_quarterly_with_states.csv", index=False)
    print("\nSaved 'decoded_quarterly_with_states.csv' with Viterbi states.")

    # (optional) save params for reuse
    params = {
        "pi": pi.tolist(),
        "A": A.tolist(),
        "lambdas": lambdas.tolist(),
        "mus": mus.tolist(),
        "sigmas": sigmas.tolist(),
        "loglik_history": [float(x) for x in ll_hist]
    }
    with open("hmm_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print("Saved fitted params -> hmm_params.json")

## example usage: python3 joint_hmm_em.py
