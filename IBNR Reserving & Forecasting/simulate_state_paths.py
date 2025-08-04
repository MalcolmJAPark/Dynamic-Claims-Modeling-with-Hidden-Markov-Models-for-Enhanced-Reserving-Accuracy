# simulate_state_paths.py

import numpy as np

def sample_initial_states(pi_T, N):
    """
    Sample initial latent states at time T for N Monte Carlo paths.
    
    Parameters:
    - pi_T: array-like of shape (2,) giving P(S_T = [0,1] | data)
    - N: number of simulated paths
    
    Returns:
    - initial_states: numpy array of shape (N,) with values 0 or 1
    """
    return np.random.choice([0, 1], size=N, p=pi_T)

def simulate_state_paths(A, initial_states, H):
    """
    Simulate state trajectories for H future steps, given transition matrix A.
    
    Parameters:
    - A: 2x2 transition matrix, A[i,j] = P(S_{t+1}=j | S_t=i)
    - initial_states: array of shape (N,) giving state at time T for each path
    - H: number of future quarters to simulate
    
    Returns:
    - states: numpy array of shape (N, H+1) of simulated states,
              where column 0 is the initial state at time T
    """
    N = initial_states.shape[0]
    states = np.zeros((N, H+1), dtype=int)
    states[:, 0] = initial_states

    for h in range(1, H+1):
        prev = states[:, h-1]
        probs = A[prev]       # shape (N, 2)
        # sample next state: for each path i, choose 1 if u < P(prev->1)
        u = np.random.rand(N)
        states[:, h] = (u < probs[:, 1]).astype(int)
    return states

def empirical_transition_matrix(states):
    """
    Compute empirical transition frequencies from simulated state paths.
    
    Parameters:
    - states: numpy array of shape (N, H+1)
    
    Returns:
    - emp_A: 2x2 numpy array of empirical P(i->j)
    """
    N, H_plus_1 = states.shape
    H = H_plus_1 - 1
    counts = np.zeros((2, 2), dtype=int)
    
    for t in range(H):
        from_states = states[:, t]
        to_states = states[:, t+1]
        for i in [0, 1]:
            idx = (from_states == i)
            # count transitions from i to j
            for j in [0, 1]:
                counts[i, j] += np.sum(to_states[idx] == j)
    
    # normalize rows
    emp_A = counts / counts.sum(axis=1, keepdims=True)
    return emp_A

if __name__ == "__main__":
    # Example usage:
    # Replace these with your EM-fitted filtered probabilities and transition matrix
    pi_T = np.array([0.9982, 0.0018])  # P(S_T = low, high | data)
    A    = np.array([[0.87804812, 0.12195188],
                     [0.82820302, 0.17179698]])
    
    # Simulation parameters
    N = 1000   # number of Monte Carlo paths
    H = 4      # quarters to project
    
    # 1. Sample initial states at time T
    initial_states = sample_initial_states(pi_T, N)
    
    # 2. Simulate forward H steps
    states = simulate_state_paths(A, initial_states, H)
    
    # 3. Compute empirical transition matrix
    emp_A = empirical_transition_matrix(states)
    
    # 4. Output results
    print("Target transition matrix A:")
    print(A)
    print("\nEmpirical transition matrix from simulation:")
    print(emp_A)

## example usage: python3 simulate_state_paths.py
## for custom case, replace value for pi_T and A with fitted values: A - transition matrix from joint_hmm_em.py,
## and pi_T - filtered posterior at the cut off date from posterior_at_asofdate.py.
