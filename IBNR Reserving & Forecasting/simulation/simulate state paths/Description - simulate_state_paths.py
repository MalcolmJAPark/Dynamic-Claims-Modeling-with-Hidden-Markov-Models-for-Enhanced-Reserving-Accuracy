simulate_state_paths.py
Simulates latent-state trajectories **beyond the cut-off quarter T** for a 2-state HMM (e.g., low-risk = 0, high-risk = 1). The script:

* **Loads**:

  * Transition matrix `A` from `hmm_params.json`
  * Filtered posterior at time **T**, `pi_T = [P(S_T=0), P(S_T=1)]`, from `posterior_at_T.json`
* **Samples** the initial states at **T** from `pi_T`
* **Evolves** each path forward **H** quarters via the Markov transition `A`
* **Reports** an empirical transition matrix from the simulated paths to sanity-check that sampling matches `A`
* **Saves** the simulated state paths for downstream simulations of counts/severity/reserves

## Inputs

* `hmm_params.json` — must contain key `"A"` as a 2×2 list/array. Rows are from-state, columns are to-state. Rows are defensively normalized.
* `posterior_at_T.json` — must contain:

  * `posterior_T.low` and `posterior_T.high` (float probabilities summing to 1 after normalization)
  * Optional `as_of_quarter` (string) for logging

## Outputs

* `states_paths.npy` — NumPy array of shape **(N, H+1)** with integer states in `{0,1}`; **column 0 is S\_T**
* `states_paths.csv` (optional with `--save-csv`) — long format with columns: `path, horizon, state`
* Console printout:

  * As-of quarter label (if provided)
  * Target transition matrix `A`
  * Empirical transition matrix estimated from simulated transitions

## Usage

```bash
python simulate_state_paths.py N H [--seed SEED] [--params hmm_params.json] [--posterior posterior_at_T.json] [--save-csv]
```

**Example:**

```bash
python simulate_state_paths.py 1000 4 --seed 2025 --save-csv
```

## Notes

* Uses NumPy’s `Generator` for reproducible RNG via `--seed`.
* Assumes a **2-state** HMM; extend shapes if you later add more states.
* The empirical transition matrix is a quick health check: with large `N` and `H`, it should be close to `A`.
