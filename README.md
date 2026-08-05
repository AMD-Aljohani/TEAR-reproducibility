# Online Resource 1 – TEAR Reproducibility Archive

**Manuscript**

*Endpoint and Dynamic-Parameter Inference for a Truncated-Exponential Autoregressive Markov Chain*

**Authors:** Abdulaziz M. D. Aljohani and Khaled M. Alhawiti

**Affiliation:** University of Tabuk, Tabuk, Saudi Arabia

**Corresponding author:** Abdulaziz M. D. Aljohani

**E-mail:** a-aljohani@ut.edu.sa

---

## Overview

This repository contains the complete reproducibility archive accompanying the manuscript

> **Endpoint and Dynamic-Parameter Inference for a Truncated-Exponential Autoregressive Markov Chain**

The archive includes:

- complete source code;
- simulation and estimation programs;
- figure-generation scripts;
- reproducibility utilities;
- processed numerical results used in the manuscript;
- deterministic workload-reproduction pipeline;
- automated validation tests.

The repository is intended to allow independent verification of every reported numerical result.

---

## Repository Contents

- **`code/`**  
  Exact-kernel TEAR simulation and estimation, endpoint experiments, boundary and misspecification studies, numerical-tolerance and indexing verification, figure generation, summary generation, automated validation tests, and workload-application reproduction scripts.

- **`results/`**  
  Raw and summarized simulation outputs, constraint-face audits, endpoint finite-sample studies, wall-density analyses, and locked application summaries used in the manuscript.

- **`figures/`**  
  Publication-quality figures together with diagnostic graphics.

- **`SOFTWARE.txt`**  
  Software environment used to generate the reported results.

- **`SHA256SUMS.txt`**  
  SHA-256 integrity manifest for the archive.

---

# Quick Validation

From the repository root run

```bash
python code/test_step3.py
```

Expected output

```text
11/11 tests passed
```

The automated tests verify:

- inverse-CDF simulation;
- conditional moments;
- analytic likelihood gradient;
- parameter recovery;
- post-initial endpoint convention;
- wall-density calculation;
- distinction between numerical optimizer termination and genuine constraint-face solutions;
- exact-strip monotonicity;
- finite-sample endpoint-tail verification;
- timestamp handling for both Unix-second and genuine millisecond archives.

To regenerate all derived summaries, figures, validation outputs, and refresh the integrity manifest run

```bash
python code/rebuild_v5_derived_outputs.py
```

The complete Monte Carlo study can be regenerated using the execution modes documented in

```
code/run_step3.py
```

The full simulation campaign is substantially longer than the automated validation suite.

---

# Endpoint Convention

All joint endpoint calculations use

```text
Khat_n = max(X_1, ..., X_n)
```

together with the trimmed conditional likelihood based on transitions

```text
X_t → X_{t+1},    t = 1,...,n−1.
```

The arbitrary initial state

```text
X_0
```

is excluded from the endpoint maximum.

---

# Positive-Rate Restriction

The simulations estimate the decreasing-density TEAR subfamily in wall-rate coordinates

```text
r(x) = q + gamma (K − x),

q ≥ delta,
gamma ≥ 0.
```

The truncated-exponential density remains mathematically valid for signed rates, including the uniform-density limit when the rate approaches zero.

Consequently, the lower bound on `q` is a modeling restriction rather than a normalization requirement.

Correct-specification simulations use

```text
delta = 0.02
```

whereas the workload application uses the manuscript's application setting, configurable through

```text
--delta
```

---

# Workload Application Reproduction

The Bitbrains GWA-T-12 **fastStorage** workload archive is publicly available but is **not redistributed** in this repository.

The complete deterministic preprocessing, screening, model fitting, PIT diagnostics, chronological holdout evaluation, and focal-series Nyström analysis can be reproduced using either

```bash
python code/reproduce_application.py --download --out results/application_reproduced
```

or

```bash
python code/reproduce_application.py \
    --archive /path/to/gwa_t_12_fastStorage.zip \
    --out results/application_reproduced
```

The official download URL is embedded directly in the script.

The reproduction writes

```text
locked_summary_comparison.csv
```

which compares every reproduced quantity with the locked manuscript summaries contained in

```text
results/application_*_summary.csv
```

The complete application pipeline was independently rerun twice on **4 August 2026** against the official public archive (SHA-256 recorded in

```text
results/application_archive_provenance.csv
```

).

Both validation runs

- screened 65 workload series,
- retained VM 718 as the focal example, and
- reproduced all 34 monitored population, predictive, and focal-series quantities with zero numerical differences.

---

# Included Reproducibility Components

This repository includes

- exact-kernel TEAR simulation and estimation software;
- known-wall and estimated-wall plug-in invariance analyses;
- complete correct-specification Monte Carlo studies;
- constraint-face frequency analyses;
- interior-only coverage evaluations;
- Monte Carlo standard errors;
- finite-sample wall-density bias studies;
- feasible endpoint-interval coverage analyses;
- deterministic public-workload reproduction scripts;
- timestamp validation supporting both Unix-second and millisecond archive formats;
- exact upper-strip probability verification;
- finite-sample endpoint-tail bound verification;
- explicit wall-density bracket verification;
- exact binomial confidence interval calculations;
- eleven automated validation tests;
- validated 65-series workload reproduction together with the locked summary tables used in the manuscript.

---

# Citation

If you use this repository, please cite the accompanying research article.

A BibTeX entry and persistent DOI will be added after Zenodo archival and updated after journal publication.

---

# License

This repository is distributed under the **BSD 3-Clause License**.

See the accompanying `LICENSE` file for details.
