# strict_ibnr_at_T.py
import os, json, argparse
import numpy as np
import pandas as pd
from scipy.special import logsumexp, gammaln

# ---------------- RNG ----------------
def rng(seed=2025):
    return np.random.default_rng(seed)

# ---------------- Emission helpers ----------------
def log_poisson_pmf(k, lam):
    lam = np.maximum(lam, 1e-8)
    return k * np.log(lam) - lam - gammaln(k + 1)

def log_lognormal_pdf(x, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    return -np.log(x * sigma * np.sqrt(2*np.pi) + 1e-16) - ((np.log(x) - mu)**2) / (2 * sigma**2)

def build_log_emissions(counts, logsev, sev_mask, lambdas, mus, sigmas):
    T = len(counts); K = len(lambdas)
    log_b = np.zeros((T, K))
    x_obs = np.exp(logsev[sev_mask])
    for k in range(K):
        # frequency for all quarters
        log_b[:, k] = log_poisson_pmf(counts, lambdas[k])
        # severity term only where observed
        sev_term_obs = log_lognormal_pdf(x_obs, mus[k], sigmas[k])
        tmp = np.zeros(T)
        tmp[sev_mask] = sev_term_obs
        log_b[:, k] += tmp
    return log_b

# ---------------- EM (only if we lack params) ----------------
def e_step(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas):
    T, K = len(counts), len(pi)
    log_b = build_log_emissions(counts, logsev, sev_mask, lambdas, mus, sigmas)
    # forward
    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(pi + 1e-16) + log_b[0]
    for t in range(1, T):
        log_alpha[t] = log_b[t] + logsumexp(log_alpha[t-1][:, None] + np.log(A + 1e-16), axis=0)
    ll = logsumexp(log_alpha[-1])
    # backward
    log_beta = np.zeros((T, K))
    for t in range(T-2, -1, -1):
        log_beta[t] = logsumexp(np.log(A + 1e-16) + (log_b[t+1] + log_beta[t+1])[None, :], axis=1)
    # posteriors
    log_gamma = log_alpha + log_beta
    gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        m = (log_alpha[t][:, None] + np.log(A + 1e-16) + log_b[t+1][None, :] + log_beta[t+1][None, :])
        xi[t] = np.exp(m - logsumexp(m))
    return gamma, xi, ll

def m_step(counts, logsev, sev_mask, gamma, xi):
    T, K = gamma.shape
    pi_new = gamma[0].copy()
    A_new = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-16)
    A_new = A_new / A_new.sum(axis=1, keepdims=True)

    lambdas_new = (gamma * counts[:, None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-16)
    lambdas_new = np.maximum(lambdas_new, 1e-8)

    g_obs = gamma[sev_mask]
    y_obs = logsev[sev_mask]
    denom = g_obs.sum(axis=0) + 1e-16
    mus_new = (g_obs * y_obs[:, None]).sum(axis=0) / denom
    sigmas_new = np.sqrt(((g_obs * (y_obs[:, None] - mus_new)**2).sum(axis=0)) / denom)
    sigmas_new = np.maximum(sigmas_new, 1e-6)
    return pi_new, A_new, lambdas_new, mus_new, sigmas_new

def fit_em(counts, logsev, sev_mask, max_iter=200, tol=1e-4, seed=42):
    np.random.seed(seed)
    mean_c = counts.mean()
    obs = logsev[sev_mask]
    mls, sls = (obs.mean() if obs.size else 0.0), (obs.std() if obs.size else 1.0)
    pi = np.array([0.5, 0.5])
    A  = np.array([[0.9, 0.1],[0.1, 0.9]])
    lambdas = np.array([0.5*mean_c, 1.5*mean_c]).clip(1e-8, None)
    mus     = np.array([mls - 0.3*sls, mls + 0.3*sls])
    sigmas  = np.array([sls, sls]).clip(1e-6, None)

    ll_hist=[]
    for it in range(max_iter):
        gamma, xi, ll = e_step(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas)
        ll_hist.append(ll)
        if len(ll_hist) >= 2 and abs(ll_hist[-1]-ll_hist[-2]) < tol:
            print(f"EM converged at iter {it}, ll={ll:.4f}")
            break
        pi, A, lambdas, mus, sigmas = m_step(counts, logsev, sev_mask, gamma, xi)
    return (pi, A, lambdas, mus, sigmas), ll_hist

# ---------------- Viterbi ----------------
def viterbi(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas):
    T, K = len(counts), len(pi)
    log_b = build_log_emissions(counts, logsev, sev_mask, lambdas, mus, sigmas)
    delta = np.zeros((T, K)); psi = np.zeros((T, K), dtype=int)
    delta[0] = np.log(pi + 1e-16) + log_b[0]
    for t in range(1, T):
        for k in range(K):
            seq = delta[t-1] + np.log(A[:, k] + 1e-16)
            psi[t, k] = np.argmax(seq)
            delta[t, k] = seq[psi[t, k]] + log_b[t, k]
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]
    return states

# ---------------- Strict IBNR core ----------------
def tail_prob(delay_ds, delay_ps, d_star):
    return delay_ps[delay_ds > d_star].sum()

def conditional_tail_draws(RNG, delay_ds, delay_ps, d_star, size):
    mask = delay_ds > d_star
    if not np.any(mask):
        return np.empty(0, dtype=int)
    ps = delay_ps[mask]; ps = ps / ps.sum()
    return RNG.choice(delay_ds[mask], size=size, p=ps)

def simulate_strict_ibnr_at_T(RNG, counts, states, lambdas, mus, sigmas,
                              delay_ds, delay_ps, r=0.0, N_paths=5000):
    """
    Strict IBNR at valuation quarter T.
    For each historical quarter t <= T:
      U_t ~ Binomial(n = counts[t], p = P(L > T - t))
      draw U_t severities ~ state_t distribution
      discount by extra delay beyond T if r>0
    """
    T = len(states) - 1
    max_delay = int(delay_ds.max())
    t_range = range(max(0, T - max_delay), T + 1)

    reserves = np.zeros(N_paths)
    for i in range(N_paths):
        total = 0.0
        for t in t_range:
            d_star = T - t
            p_tail = tail_prob(delay_ds, delay_ps, d_star)
            if p_tail <= 0.0 or counts[t] <= 0:
                continue
            # Binomial thinning of observed counts at t
            U_t = RNG.binomial(n=int(counts[t]), p=p_tail)
            if U_t <= 0:
                continue

            # delays conditional on > d_star to compute discount beyond T
            Ls = conditional_tail_draws(RNG, delay_ds, delay_ps, d_star, size=U_t)
            extra = Ls - d_star  # quarters beyond T

            # draw severities from state at t
            s_t = states[t]
            losses = RNG.lognormal(mean=mus[s_t], sigma=sigmas[s_t], size=U_t)
            if r > 0:
                total += np.sum(losses * (1.0 + r) ** (-extra))
            else:
                total += np.sum(losses)
        reserves[i] = total
    return reserves

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser(description="Strict IBNR at valuation quarter T")
    p.add_argument("--agg", default="aggregated_quarterly.csv")
    p.add_argument("--params", default="hmm_params.json", help="fitted params (optional)")
    p.add_argument("--states", default="decoded_quarterly_with_states.csv", help="decoded states CSV (optional)")
    p.add_argument("-N", "--npaths", type=int, default=5000)
    p.add_argument("-r", "--rate", type=float, default=0.0, help="per-quarter discount rate")
    p.add_argument("--delay-ds", type=int, nargs="*", default=[0,1,2,3,4])
    p.add_argument("--delay-ps", type=float, nargs="*", default=[0.50,0.30,0.10,0.07,0.03])
    p.add_argument("--seed", type=int, default=2025)
    args = p.parse_args()

    RNG = rng(args.seed)

    # Load aggregated data
    df = pd.read_csv(args.agg, parse_dates=["quarter_start"]).sort_values("quarter_start")
    counts = df["n_claims"].astype(int).to_numpy()
    logsev = df["avg_log_severity"].to_numpy(float)
    sev_mask = ~np.isnan(logsev)
    T_idx = len(counts) - 1
    print(f"As-of quarter (T): {df.iloc[T_idx]['quarter_start']:%Y-%m-%d}")

    # Load params if available, else fit
    if os.path.exists(args.params):
        with open(args.params, "r") as f:
            P = json.load(f)
        pi = np.asarray(P["pi"], float)
        A  = np.asarray(P["A"], float); A = A / A.sum(axis=1, keepdims=True)
        lambdas = np.asarray(P["lambdas"], float)
        mus     = np.asarray(P["mus"], float)
        sigmas  = np.asarray(P["sigmas"], float)
        print("Loaded HMM params from hmm_params.json")
    else:
        print("Params not found; fitting EM…")
        (pi, A, lambdas, mus, sigmas), _ = fit_em(counts, logsev, sev_mask)

    # Load states if available, else Viterbi-decode using current params
    if os.path.exists(args.states):
        st = pd.read_csv(args.states)
        # Assume same order; else merge on quarter_start
        states = st["state"].astype(int).to_numpy()
        print("Loaded decoded states from decoded_quarterly_with_states.csv")
    else:
        print("States not found; decoding with Viterbi…")
        states = viterbi(counts, logsev, sev_mask, pi, A, lambdas, mus, sigmas)

    # Normalize delay pmf
    delay_ds = np.asarray(args.delay_ds, int)
    delay_ps = np.asarray(args.delay_ps, float)
    delay_ps = delay_ps / delay_ps.sum()

    # Simulate strict IBNR at T
    reserves = simulate_strict_ibnr_at_T(
        RNG, counts, states, lambdas, mus, sigmas,
        delay_ds, delay_ps, r=args.rate, N_paths=args.npaths
    )

    # Summaries + save
    qs = [0.50, 0.75, 0.90, 0.95, 0.99]
    qv = np.quantile(reserves, qs)
    print("\nStrict IBNR at T (PV) percentiles:")
    for pctl, val in zip(qs, qv):
        print(f"  {int(pctl*100)}th: {val:,.0f}")

    out_csv = "strict_ibnr_reserves.csv"
    pd.DataFrame({"reserve_pv": reserves}).to_csv(out_csv, index=False)
    print(f"\nSaved path-level reserves -> {out_csv}")

if __name__ == "__main__":
    main()

## example usage: python3 strict_ibnr_at_T.py
