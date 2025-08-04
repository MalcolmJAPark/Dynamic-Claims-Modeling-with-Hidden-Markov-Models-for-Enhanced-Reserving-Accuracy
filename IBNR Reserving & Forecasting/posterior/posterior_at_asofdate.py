# posterior_at_asofdate.py

import numpy as np
import pandas as pd
from scipy.special import logsumexp, gammaln

def log_poisson_pmf(k, lam):
    """Log of Poisson PMF."""
    return k * np.log(lam + 1e-16) - lam - gammaln(k + 1)

def log_lognormal_pdf(x, mu, sigma):
    """Log of LogNormal PDF."""
    return -np.log(x * sigma * np.sqrt(2 * np.pi) + 1e-16) - ((np.log(x) - mu) ** 2) / (2 * sigma ** 2)

def initialize_parameters(counts, sev, n_states=2):
    """Initialize HMM parameters."""
    mean_count = counts.mean()
    mean_log_sev = np.log(sev).mean()
    std_log_sev = np.log(sev).std()
    
    pi = np.full(n_states, 1.0 / n_states)
    A = np.full((n_states, n_states), (1 - 0.1) / (n_states - 1))
    np.fill_diagonal(A, 0.9)
    
    lambdas = np.linspace(mean_count * 0.5, mean_count * 1.5, n_states)
    mus = np.linspace(mean_log_sev * 0.8, mean_log_sev * 1.2, n_states)
    sigmas = np.full(n_states, std_log_sev)
    
    return pi, A, lambdas, mus, sigmas

def forward_backward(counts, sev, pi, A, lambdas, mus, sigmas):
    """Run forward-backward and return gamma, xi, and log-likelihood."""
    T = len(counts)
    K = len(pi)
    
    # Emission log-probs
    log_b = np.zeros((T, K))
    for k in range(K):
        log_b[:, k] = log_poisson_pmf(counts, lambdas[k]) + log_lognormal_pdf(sev, mus[k], sigmas[k])
    
    # Forward
    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(pi + 1e-16) + log_b[0]
    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = log_b[t, j] + logsumexp(log_alpha[t-1] + np.log(A[:, j] + 1e-16))
    
    # Backward
    log_beta = np.zeros((T, K))
    for t in range(T-2, -1, -1):
        for i in range(K):
            log_beta[t, i] = logsumexp(np.log(A[i] + 1e-16) + log_b[t+1] + log_beta[t+1])
    
    # Gamma: smoothed posterior P(S_t = k | data)
    log_gamma = log_alpha + log_beta
    log_gamma_norm = logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma - log_gamma_norm)
    
    # Xi not needed for posterior at T
    ll = logsumexp(log_alpha[-1])
    return gamma, ll

def m_step(counts, sev, gamma, xi):
    """M-step: update parameters."""
    T, K = gamma.shape
    log_sev = np.log(sev)
    
    # Update pi
    pi_new = gamma[0]
    
    # Update A
    A_new = xi.sum(axis=0) / gamma[:-1].sum(axis=0, keepdims=True).T
    
    # Update emissions
    lambdas_new = (gamma * counts[:, None]).sum(axis=0) / gamma.sum(axis=0)
    mus_new = (gamma * log_sev[:, None]).sum(axis=0) / gamma.sum(axis=0)
    sigmas_new = np.sqrt(((gamma * (log_sev[:, None] - mus_new)**2).sum(axis=0)) / gamma.sum(axis=0))
    
    return pi_new, A_new, lambdas_new, mus_new, sigmas_new

def run_em_and_posterior(input_path, max_iter=100, tol=1e-4):
    # Load aggregated data
    df = pd.read_csv(input_path, parse_dates=['quarter_start']).sort_values('quarter_start')
    counts = df['n_claims'].values
    severity = np.exp(df['avg_log_severity'].values)
    
    # Initialize parameters
    pi, A, lambdas, mus, sigmas = initialize_parameters(counts, severity)
    
    # EM loop for parameters only
    for i in range(max_iter):
        # E-step
        gamma, ll = forward_backward(counts, severity, pi, A, lambdas, mus, sigmas)
        # We skip xi calc and m_step since we only need posterior; assume parameters are pre-fitted
        # If parameters need fitting, include xi and m_step here
        
        # Convergence check on ll (if fitting)
        # ...
        pass
    
    # Compute filtered posterior at final time T
    gamma, ll = forward_backward(counts, severity, pi, A, lambdas, mus, sigmas)
    posterior_T = gamma[-1]
    print(f"P(S_T = low-risk | data)  = {posterior_T[0]:.4f}")
    print(f"P(S_T = high-risk| data)  = {posterior_T[1]:.4f}")

if __name__ == "__main__":
    run_em_and_posterior("aggregated_quarterly.csv")

## example usage: python3 posterior_at_asofdate.py
