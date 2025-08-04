# joint_hmm_em.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp, gammaln

def log_poisson_pmf(k, lam):
    """Log of Poisson PMF using gammaln for factorial."""
    # log P(K=k) = k*log(lam) - lam - log(k!)
    return k * np.log(lam + 1e-16) - lam - gammaln(k + 1)

def log_lognormal_pdf(x, mu, sigma):
    """Log of LogNormal PDF."""
    return -np.log(x * sigma * np.sqrt(2 * np.pi) + 1e-16) - ((np.log(x) - mu) ** 2) / (2 * sigma ** 2)

def initialize_parameters(n_states, counts, sev, random_state=42):
    np.random.seed(random_state)
    mean_count = counts.mean()
    mean_log_sev = np.log(sev).mean()
    std_log_sev = np.log(sev).std()
    
    # Uniform initial state distribution
    pi = np.full(n_states, 1.0 / n_states)
    # Transition matrix with self-loop bias
    A = np.full((n_states, n_states), (1 - 0.1) / (n_states - 1))
    np.fill_diagonal(A, 0.9)
    
    # Emission parameters
    lambdas = np.linspace(mean_count * 0.5, mean_count * 1.5, n_states)
    mus = np.linspace(mean_log_sev * 0.8, mean_log_sev * 1.2, n_states)
    sigmas = np.full(n_states, std_log_sev)
    
    return pi, A, lambdas, mus, sigmas

def forward_backward(counts, sev, pi, A, lambdas, mus, sigmas):
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
    for t in range(T - 2, -1, -1):
        for i in range(K):
            log_beta[t, i] = logsumexp(np.log(A[i] + 1e-16) + log_b[t+1] + log_beta[t+1])

    # Gamma
    log_gamma = log_alpha + log_beta
    log_gamma_norm = logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma - log_gamma_norm)

    # Xi
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        num = (log_alpha[t][:, None] +
               np.log(A + 1e-16) +
               log_b[t+1][None, :] +
               log_beta[t+1][None, :])
        denom = logsumexp(num)
        xi[t] = np.exp(num - denom)

    # Log-likelihood
    ll = logsumexp(log_alpha[-1])
    return gamma, xi, ll

def m_step(counts, sev, gamma, xi):
    log_sev = np.log(sev)
    T, K = gamma.shape

    pi_new = gamma[0]
    A_new = xi.sum(axis=0) / gamma[:-1].sum(axis=0, keepdims=True).T

    lambdas_new = (gamma * counts[:, None]).sum(axis=0) / gamma.sum(axis=0)
    mus_new = (gamma * log_sev[:, None]).sum(axis=0) / gamma.sum(axis=0)
    sigmas_new = np.sqrt(((gamma * (log_sev[:, None] - mus_new)**2).sum(axis=0)) / gamma.sum(axis=0))

    return pi_new, A_new, lambdas_new, mus_new, sigmas_new

def run_em(data_path, max_iter=100, tol=1e-4):
    df = pd.read_csv(data_path, parse_dates=['quarter_start']).sort_values('quarter_start')
    counts = df['n_claims'].values
    severity = np.exp(df['avg_log_severity'].values)

    pi, A, lambdas, mus, sigmas = initialize_parameters(2, counts, severity)
    log_likelihoods = []

    for i in range(max_iter):
        gamma, xi, ll = forward_backward(counts, severity, pi, A, lambdas, mus, sigmas)
        log_likelihoods.append(ll)
        if i and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print(f"Converged at iteration {i}, log-likelihood = {ll:.4f}")
            break
        pi, A, lambdas, mus, sigmas = m_step(counts, severity, gamma, xi)

    print("\nFinal parameters:")
    print(f"pi = {pi}")
    print(f"A =\n{A}")
    for k in range(len(pi)):
        print(f"State {k}: λ = {lambdas[k]:.4f}, μ = {mus[k]:.4f}, σ = {sigmas[k]:.4f}")

    plt.figure()
    plt.plot(log_likelihoods, marker='o')
    plt.title("Joint HMM EM Log-Likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.tight_layout()
    plt.savefig("joint_hmm_em_loglikelihood.png")
    plt.close()
    print("Saved 'joint_hmm_em_loglikelihood.png'.")

if __name__ == "__main__":
    run_em("aggregated_quarterly.csv")


## example usage: python3 joint_hmm_em.py
