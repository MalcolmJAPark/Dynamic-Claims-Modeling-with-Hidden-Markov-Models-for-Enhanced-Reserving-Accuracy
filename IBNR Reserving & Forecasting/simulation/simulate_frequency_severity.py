# simulate_frequency_severity.py

import numpy as np
import pandas as pd

def sample_initial_states(pi_T, N):
    """Sample initial latent states at time T for N Monte Carlo paths."""
    return np.random.choice([0, 1], size=N, p=pi_T)

def simulate_state_paths(A, initial_states, H):
    """Simulate state trajectories for H future steps."""
    N = initial_states.shape[0]
    states = np.zeros((N, H+1), dtype=int)
    states[:, 0] = initial_states
    for h in range(1, H+1):
        prev = states[:, h-1]
        probs = A[prev]       # shape (N, 2)
        u = np.random.rand(N)
        states[:, h] = (u < probs[:, 1]).astype(int)
    return states

def simulate_claims(states, lambdas, mus, sigmas, severity_dist='lognormal'):
    """
    Given simulated states, draw counts and severities.
    - severity_dist: 'lognormal' or 'gamma'
    """
    N, H1 = states.shape
    counts = np.zeros((N, H1), dtype=int)
    losses = np.zeros((N, H1), dtype=float)
    
    for i in range(N):
        for h in range(1, H1):
            s = states[i, h]
            # Frequency draw
            lam = lambdas[s]
            n_claims = np.random.poisson(lam)
            counts[i, h] = n_claims
            
            # Severity draws
            if n_claims > 0:
                if severity_dist == 'lognormal':
                    sev_draws = np.random.lognormal(mean=mus[s], sigma=sigmas[s], size=n_claims)
                else:  # gamma
                    # mus and sigmas interpreted as shape and scale for gamma
                    sev_draws = np.random.gamma(shape=mus[s], scale=sigmas[s], size=n_claims)
                losses[i, h] = sev_draws.sum()
    return counts, losses

if __name__ == "__main__":
    # Example parameters (replace with your fitted values)
    pi_T = np.array([0.9982, 0.0018])
    A = np.array([[0.87804812, 0.12195188],
                  [0.82820302, 0.17179698]])
    lambdas = np.array([2.5373, 27.6194])
    mus = np.array([8.7550, 9.8738])       # for lognormal: means of log(X)
    sigmas = np.array([1.4324, 0.8136])    # for lognormal: std dev of log(X)
    
    # Simulation settings
    N = 5000   # Monte Carlo paths
    H = 8      # quarters beyond cut-off
    
    # 1. Sample initial states
    initial_states = sample_initial_states(pi_T, N)
    
    # 2. Simulate state trajectories
    states = simulate_state_paths(A, initial_states, H)
    
    # 3. Draw counts and severities
    counts, losses = simulate_claims(states, lambdas, mus, sigmas, severity_dist='lognormal')
    
    # 4. Aggregate results to a DataFrame
    records = []
    for i in range(N):
        for h in range(1, H+1):
            records.append({
                'path': i,
                'horizon': h,
                'state': states[i, h],
                'n_claims': counts[i, h],
                'total_loss': losses[i, h]
            })
    df_sim = pd.DataFrame.from_records(records)
    
    # 5. Save to CSV
    df_sim.to_csv('simulated_claims.csv', index=False)
    print(f"Simulation complete: {N} paths × {H} quarters → 'simulated_claims.csv'")


## example usage: python3 simulate_frequency_severity.py