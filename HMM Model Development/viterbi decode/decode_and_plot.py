# decode_and_plot.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp, gammaln

# ----- Emission log-prob functions -----
def log_poisson_pmf(k, lam):
    return k * np.log(lam + 1e-16) - lam - gammaln(k + 1)

def log_lognormal_pdf(x, mu, sigma):
    return -np.log(x * sigma * np.sqrt(2*np.pi) + 1e-16) \
           - ((np.log(x) - mu)**2)/(2*sigma**2)

# ----- EM (forward-backward & M-step) to fit params -----
def initialize_parameters(counts, sev):
    mean_c, mean_ls = counts.mean(), np.log(sev).mean()
    std_ls = np.log(sev).std()
    # 2 states
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9,0.1],[0.1,0.9]])
    lambdas = np.linspace(mean_c*0.5, mean_c*1.5, 2)
    mus      = np.linspace(mean_ls*0.8, mean_ls*1.2, 2)
    sigmas   = np.array([std_ls, std_ls])
    return pi, A, lambdas, mus, sigmas

def forward_backward(counts, sev, pi, A, lambdas, mus, sigmas):
    T, K = len(counts), len(pi)
    # log emission probs
    log_b = np.vstack([
        log_poisson_pmf(counts, lam) + log_lognormal_pdf(sev, mu, sig)
        for lam,mu,sig in zip(lambdas, mus, sigmas)
    ]).T  # shape (T,K)

    # alpha
    log_alpha = np.zeros((T,K))
    log_alpha[0] = np.log(pi+1e-16) + log_b[0]
    for t in range(1,T):
        for j in range(K):
            log_alpha[t,j] = log_b[t,j] + logsumexp(log_alpha[t-1] + np.log(A[:,j]+1e-16))

    # beta
    log_beta = np.zeros((T,K))
    for t in range(T-2,-1,-1):
        for i in range(K):
            log_beta[t,i] = logsumexp(np.log(A[i]+1e-16) + log_b[t+1] + log_beta[t+1])

    # gamma
    log_gamma = log_alpha + log_beta
    norm = logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma - norm)

    # xi
    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        num = (log_alpha[t][:,None]
             + np.log(A+1e-16)
             + log_b[t+1][None,:]
             + log_beta[t+1][None,:])
        xi[t] = np.exp(num - logsumexp(num))

    ll = logsumexp(log_alpha[-1])
    return gamma, xi, ll

def m_step(counts, sev, gamma, xi):
    T,K = gamma.shape
    log_sev = np.log(sev)
    # π
    pi_new = gamma[0]
    # A
    A_new = xi.sum(0) / gamma[:-1].sum(0)[:,None]
    # emissions
    lambdas_new = (gamma*counts[:,None]).sum(0)/gamma.sum(0)
    mus_new      = (gamma*log_sev[:,None]).sum(0)/gamma.sum(0)
    sigmas_new   = np.sqrt(((gamma*(log_sev[:,None]-mus_new)**2).sum(0))/gamma.sum(0))
    return pi_new, A_new, lambdas_new, mus_new, sigmas_new

def fit_em(counts, sev, max_iter=100, tol=1e-4):
    pi,A,lambdas,mus,sigmas = initialize_parameters(counts, sev)
    ll_hist=[]
    for i in range(max_iter):
        gamma, xi, ll = forward_backward(counts, sev, pi, A, lambdas, mus, sigmas)
        ll_hist.append(ll)
        if i>0 and abs(ll_hist[-1]-ll_hist[-2])<tol:
            print(f"EM converged at iter {i}, ll={ll:.4f}")
            break
        pi,A,lambdas,mus,sigmas = m_step(counts, sev, gamma, xi)
    return pi,A,lambdas,mus,sigmas,ll_hist

# ----- Viterbi decoding -----
def viterbi(counts, sev, pi, A, lambdas, mus, sigmas):
    T, K = len(counts), len(pi)
    # emission logs
    log_b = np.vstack([
        log_poisson_pmf(counts, lam) + log_lognormal_pdf(sev, mu, sig)
        for lam,mu,sig in zip(lambdas, mus, sigmas)
    ]).T
    delta = np.zeros((T,K))
    psi   = np.zeros((T,K), dtype=int)

    delta[0] = np.log(pi+1e-16)+log_b[0]
    for t in range(1,T):
        for k in range(K):
            seq = delta[t-1] + np.log(A[:,k]+1e-16)
            psi[t,k] = np.argmax(seq)
            delta[t,k] = seq[psi[t,k]] + log_b[t,k]

    states = np.zeros(T, int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T-2,-1,-1):
        states[t] = psi[t+1, states[t+1]]
    return states

# ----- Main -----
if __name__=='__main__':
    # 1) Load aggregated data
    df = pd.read_csv("aggregated_quarterly.csv", parse_dates=['quarter_start'])
    df = df.sort_values('quarter_start')
    counts = df['n_claims'].values
    sev     = np.exp(df['avg_log_severity'].values)

    # 2) Fit EM
    pi,A,lam,mu,sig,ll_hist = fit_em(counts, sev)

    # 3) Decode
    states = viterbi(counts, sev, pi, A, lam, mu, sig)

    # 4) Plot counts with state coloring
    cmap = ['#1f77b4','#d62728']  # blue=low, red=high
    plt.figure(figsize=(10,5))
    plt.plot(df['quarter_start'], counts, lw=1, label='n_claims')
    plt.scatter(df['quarter_start'], counts, c=[cmap[s] for s in states], s=50, label='decoded state')
    plt.title("Quarterly Claims & Viterbi Regimes")
    plt.xlabel("Quarter")
    plt.ylabel("n_claims")
    plt.legend()
    plt.tight_layout()
    plt.savefig("viterbi_n_claims.png")
    plt.close()

    # 5) Plot avg_log_severity with state coloring
    plt.figure(figsize=(10,5))
    plt.plot(df['quarter_start'], df['avg_log_severity'], lw=1, label='avg_log_severity')
    plt.scatter(df['quarter_start'], df['avg_log_severity'], c=[cmap[s] for s in states], s=50, label='decoded state')
    plt.title("Quarterly Avg Log Severity & Viterbi Regimes")
    plt.xlabel("Quarter")
    plt.ylabel("avg_log_severity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("viterbi_avg_log_severity.png")
    plt.close()

    print("Decoded regimes plotted to 'viterbi_n_claims.png' and 'viterbi_avg_log_severity.png'.")


## example usage: python3 decode_and_plot.py
