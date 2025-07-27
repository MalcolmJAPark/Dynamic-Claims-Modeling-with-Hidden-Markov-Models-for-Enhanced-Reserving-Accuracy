# Dynamic-Claims-Modeling-with-Hidden-Markov-Models-for-Enhanced-Reserving-Accuracy
Claims Frequency-Severity Modeling Using Hidden Markov Models

## Description:
Build an Hidden Markov Model (HMM) based model to capture the **latent risk state** (e.g., safe vs. risky periods) of policyholders based on historical claims data. Use the model to forecast future claims and estimate incurred but not reported (IBNR) reserves.
* Used simulated or public auto/home insurance data with claim history per policy.
* Modeled transitions between "low-risk" and "high-risk" hidden states over time.
* Conditioned claim frequency and severity distributions on the latent state.
* Compared reserve estimates using traditional chain ladder vs. HMM-adjusted forecasts.

This project demonstrates probabilistic modeling and state estimation, reserving (IBNR estimation), and statistical inference and validation (e.g., EM algorithm and log-likelihood training)

## Project Roadmap
- [ ] Phase 1:
* Environment and tools: set up a python environment with hmmlearn, numpy, pandas, scipy, matplotlib (libraries that may be used).
- [ ] Phase 2:
* Data acquisition and preparation: download a public claims dataset (for this project, it was decided to use FIMA NFIP Redacted Claims, a dataset offered by FEMA including over 2 million redacted National Flood Insurance Program (NFIP) claims transactions, updated monthly, and is accessible for public use with tools like SQL for analysis).
[link to FIMA website API](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2)
* Data Cleaning: Parse policy-level claim counts and severities by time period (e.g., quarterly) and handle missing values, outliers; aggregate exposures if needed (benefit with using FIMA data is pre-cleaned to use but SQL is needed for use due to its large size to prevent data loss).
* Exploratory Data Analysis: Plot claim counts over time and generate visual history of severities. Compute sumamry statistics by periods and look for signs of **regime changes** (e.g., spikes in frequency and/or severity).
- [ ] Phase 3:
