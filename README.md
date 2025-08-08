# Dynamic Claims Modeling with Hidden Markov Models for Enhanced Reserving Accuracy
Claims Frequency-Severity Modeling Using Hidden Markov Models

## Overview:
This project develops a Hidden Markov Model (HMM) to capture latent risk states (e.g., safe vs. risky periods) of policyholders using historical claims data. The model is used to forecast future claims and estimate Incurred But Not Reported (IBNR) reserves.

Key features:
* Uses simulated or public auto/home insurance data with policy-level claim histories.
* Models transitions between low-risk and high-risk hidden states over time.
* Conditions claim frequency and severity distributions on the latent state.
* Compares reserve estimates from a traditional chain ladder approach with HMM-adjusted forecasts.

This project demonstrates:
* Probabilistic modeling and state estimation
* Reserving (IBNR estimation)
* Statistical inference and validation (e.g., EM algorithm, log-likelihood training)

## Project Roadmap
- [X] Phase 1:
Configure Python environment with: hmmlearn, numpy, pandas, scipy, matplotlib
- [X] Phase 2:
1. Download FIMA NFIP Redacted Claims dataset from FEMA:
  * [link to FIMA website API](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2)
  * Over 2 million redacted National Flood Insurance Program claims transactions.
2. Data Cleaning:
  * Aggregate claims to policy-level by time period (e.g., quarterly).
  * Handle missing values, outliers, and exposures.
  * Benefit: FIMA data is pre-cleaned, but SQL is recommended to handle large size efficiently.
3. Exploratory Data Analysis:
  * Plot claim counts over time.
  * Visualize severity history.
  * Compute summary statistics and identify regime changes (spikes in frequency/severity).
- [X] Phase 3:
1. State/Emission Distribution:
  * Two states: low-risk and high-risk.
  * Frequency: Poisson(λ) by state.
  * Severity: Log-Normal(μ, σ) by state.
2. HMM Implementation:
  * Prototype in Jupyter using hmmlearn or custom EM loop.
  * Fit model to time-series of counts (and optionally severity jointly).
3. Model Validation:
  * Monitor log-likelihood convergence.
  * Check logical consistency of transition matrix (e.g., low→high probability is small).
4. State Decoding:
  * Apply Viterbi algorithm to assign latent states to periods.
  * Plot decoded states vs. observed frequency/severity.
- [X] Phase 4:
1. Simulate future quarterly claim counts and severities for each latent state path.
2. Estimate IBNR reserves beyond cut-off date (asOfDate).
3. Produce predictive distributions for reserves:
  * Mean
  * Percentiles
