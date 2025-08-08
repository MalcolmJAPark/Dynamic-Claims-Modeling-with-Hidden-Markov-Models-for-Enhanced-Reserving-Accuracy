simulate_frequency_severity.py
Simulates **future claim frequencies and severities** conditional on latent state trajectories from a fitted 2-state Hidden Markov Model (HMM).
The script:

* **Seeds** the initial latent state at the cut-off quarter **T** from the filtered posterior `pi_T`
* **Simulates** state paths for `H` future quarters using the transition matrix `A`
* **Generates**:

  * Claim counts per period from a **Poisson** distribution with state-dependent rate `λ`
  * Claim severities from either a **Log-Normal** (`μ`, `σ`) or **Gamma** (shape=`μ`, scale=`σ`) distribution
* **Aggregates** counts and severities to **per-path, per-horizon totals**
* **Writes** results to `simulated_claims.csv`
* **Optionally** prints summary loss percentiles across Monte Carlo paths

---

## Inputs

* **`hmm_params.json`** — must contain:

  * `"A"` — 2×2 state transition matrix (row = from-state, col = to-state)
  * `"lambdas"` — Poisson means for each state
  * `"mus"`, `"sigmas"` — parameters for severity distribution
* **`posterior_at_T.json`** — must contain:

  * `posterior_T.low` and `posterior_T.high` (probabilities for state at T)
  * Optional `as_of_quarter` for logging

---

## Outputs

* **`simulated_claims.csv`** — table with columns:

  * `path` — Monte Carlo path index
  * `horizon` — future quarter index (1 = first quarter after T)
  * `state` — latent state at that horizon
  * `n_claims` — number of claims
  * `total_loss` — sum of severities for that period
* **Console summary** (unless `--no-summary`):

  * Percentiles (50th, 75th, 90th, 95th, 99th) of total undiscounted loss over the H quarters

---

## Usage

```bash
python simulate_frequency_severity.py N H \
    --severity {lognormal|gamma} \
    --seed SEED \
    --params hmm_params.json \
    --posterior posterior_at_T.json \
    [--no-summary]
```

**Example:**

```bash
python simulate_frequency_severity.py 5000 8 --severity lognormal --seed 2025
```

---

## Notes

* Assumes **2 latent states** (0 and 1)
* Severity model controlled by `--severity` flag:

  * `"lognormal"` — uses `mus` and `sigmas` as log-mean and log-sd
  * `"gamma"` — uses `mus` as shape and `sigmas` as scale
* Vectorized to handle large `N` efficiently
* Does **not** discount losses — see `simulate_ibnr_reserves.py` for PV computation
