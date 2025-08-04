# fit_poisson_hmm.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure you have installed hmmlearn: pip install hmmlearn
from hmmlearn.hmm import PoissonHMM

def fit_poisson_hmm(input_path: str, n_components: int = 2, n_iter: int = 100, random_state: int = 42):
    """
    Fits a 2-state Poisson HMM to the quarterly claim counts and plots convergence.
    """
    # 1. Load aggregated quarterly data
    df = pd.read_csv(input_path, parse_dates=['quarter_start'])
    df = df.sort_values('quarter_start')

    # 2. Prepare the observation sequence (n_claims) as a 2D array
    X = df['n_claims'].values.reshape(-1, 1)

    # 3. Initialize and fit the PoissonHMM
    model = PoissonHMM(
        n_components=n_components,
        n_iter=n_iter,
        tol=1e-4,
        verbose=True,
        random_state=random_state
    )
    model.fit(X)

    # 4. Extract learned parameters
    # PoissonHMM stores its rate parameters in `lambdas_` instead of `means_`
    lambdas = model.lambdas_.flatten()
    transmat = model.transmat_
    startprob = model.startprob_

    # 5. Print estimated Poisson means and transition matrix
    print("Estimated Poisson rates (λ) per state:")
    for i, lam in enumerate(lambdas):
        print(f"  State {i}: λ = {lam:.4f}")

    print("\nTransition matrix:")
    print(transmat)

    print("\nInitial state distribution:")
    print(startprob)

    # 6. Plot log-likelihood vs. EM iteration
    log_likelihoods = model.monitor_.history
    plt.figure()
    plt.plot(range(1, len(log_likelihoods) + 1), log_likelihoods)
    plt.xlabel('EM iteration')
    plt.ylabel('Log-likelihood')
    plt.title('Poisson HMM Convergence')
    plt.tight_layout()
    plt.savefig('hmm_log_likelihood.png')
    plt.close()
    print("\nConvergence plot saved as 'hmm_log_likelihood.png'.")

if __name__ == "__main__":
    fit_poisson_hmm("aggregated_quarterly.csv")



## example usage: python3 fit_poisson_hmm.py
