# Online Resource 1 - TEAR reproducibility archive

**Article:** Endpoint and Dynamic-Parameter Inference for a Truncated-Exponential Autoregressive Markov Chain  
**Authors:** Abdulaziz M. D. Aljohani and Khaled M. Alhawiti  
**Affiliation:** University of Tabuk, Tabuk, Saudi Arabia  
**Corresponding author:** Abdulaziz M. D. Aljohani  
**E-mail:** a-aljohani@ut.edu.sa  
**Target journal:** Statistical Papers

## Contents

- `code/`: exact-kernel TEAR simulation and estimation; endpoint experiments; boundary, misspecification, numerical-tolerance, and indexing checks; figure generation; v5 summary generation; automated tests; and workload-application reproduction code.
- `results/`: raw and summarized simulation outputs, constraint-face audits, endpoint finite-sample audits, and locked application summaries used in the manuscript.
- `figures/`: publication and diagnostic figures.
- `SOFTWARE.txt`: recorded software environment.
- `SHA256SUMS.txt`: archive-internal integrity manifest.

## Quick validation

From the archive root:

```bash
python code/test_step3.py
```

Expected result:

```text
11/11 tests passed
```

The tests cover the inverse-CDF simulator, conditional moments, analytic gradient, parameter recovery, the post-initial endpoint convention, wall-density calculation, and the distinction between numerical termination and a constraint-face solution.
They also verify both Unix-second timestamps used by the current public GWA-T-12 archive and genuine millisecond timestamps, despite the archive's misleading `Timestamp [ms]` header.

Regenerate the v5 audit summaries and plug-in-invariance figure, rerun the tests, and refresh the integrity manifest with:

```bash
python code/rebuild_v5_derived_outputs.py
```

The full simulation campaign can be regenerated with the modes documented in `code/run_step3.py` and the companion scripts. The complete campaign is substantially longer than the automated checks.

## Endpoint convention

All joint-endpoint calculations use

```text
Khat_n = max(X_1, ..., X_n)
```

and the trimmed conditional criterion based on transitions `X_t -> X_{t+1}` for `t=1,...,n-1`. The arbitrary initial state `X_0` is excluded from the maximum.

## Positive-rate restriction

The simulations fit the decreasing-density TEAR subfamily in wall-rate coordinates

```text
r(x) = q + gamma * (K - x),  q >= delta, gamma >= 0.
```

The truncated-exponential density is mathematically valid for signed rates, including the uniform limit at rate zero. The lower bound on `q` is therefore a modeling restriction, not a normalization requirement. Correct-specification simulations use `delta = 0.02`; the workload reproduction script uses the manuscript's application setting, configurable by `--delta`.

## Workload application reproduction

The GWA-T-12 Bitbrains `fastStorage` raw archive is public but is not redistributed. Reproduce the deterministic preprocessing, screening, TEAR and comparison fits, PIT diagnostics, chronological holdout evaluation, and focal-series Nyström calculation with either:

```bash
python code/reproduce_application.py --download --out results/application_reproduced
```

or

```bash
python code/reproduce_application.py --archive /path/to/gwa_t_12_fastStorage.zip --out results/application_reproduced
```

The official download URL is embedded in the script. The script writes `locked_summary_comparison.csv`; inspect every difference against the locked `results/application_*_summary.csv` files. The complete pipeline was rerun twice on 4 August 2026 against the official archive (SHA-256 recorded in `results/application_archive_provenance.csv`). It screened 65 series, retained VM 718 as the focal example, and reproduced all 34 monitored population, predictive, and focal-series quantities with zero differences on the validation run.

## Main V5 changes represented in the archive

- Removed the synthetic generic rate-coupling Monte Carlo figure and output.
- Added actual known-wall versus estimated-wall plug-in-invariance summaries.
- Added all correct-specification cells with constraint-face frequencies and interior-only coverage.
- Added Monte Carlo standard errors.
- Added finite-sample wall-density bias and feasible-interval coverage audits.
- Added a deterministic public-trace reproduction script.
- Corrected the public-trace reader to recognize that the current archive stores Unix seconds under a `Timestamp [ms]` header; the reader also remains compatible with millisecond timestamps.
- Added exact upper-strip probabilities, finite-sample endpoint-tail bound tables, explicit wall-density brackets, and exact binomial coverage intervals.
- Expanded automated checks to eleven, including exact-strip monotonicity, locked-data verification of the finite-sample tail sandwich, and seconds-versus-milliseconds timestamp handling.
- Replaced the earlier unexecuted 60-series application summary with the independently rerun 65-series result implied by the stated screen.
