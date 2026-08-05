# GWA-T-12 application reproduction notes

The manuscript treats the Bitbrains example as a known-wall specification stress test. The upper wall is fixed at `K=1` after conversion of CPU utilization percentages to fractions.

`code/reproduce_application.py` performs the following operations:

1. Read each VM trace from the official `fastStorage` ZIP archive.
2. Detect whether timestamps are encoded in seconds or milliseconds and select the longest run with five-minute spacing (tolerance 1 second). The current public archive labels the field `Timestamp [ms]` but stores Unix seconds.
3. Take one fixed-cadence observation every 12 records (hourly snapshots).
4. Apply the prespecified panel screen stated in the manuscript.
5. Fit the positive-rate TEAR subfamily and its independent nested model.
6. Fit a beta autoregression for the chronological 70/30 comparison.
7. Compute PIT diagnostics, boundary-mixture persistence tests, and held-out metrics.
8. Select the focal descriptive series using the stated lag-correlation and maximum rules.
9. Compute a Nyström stationary-density summary for the focal TEAR fit.
10. Write per-series and aggregate CSV files.

Because the focal series is selected partly by lag correlation and observed maximum, the manuscript does not interpret a post-selection persistence p-value or maximum-compatibility probability. Telemetry rounding and discreteness are also listed as limitations of the continuous conditional model.

The exact public archive is not redistributed. The complete pipeline was validated twice on 4 August 2026 against the official archive. The stated screen retained 65 series, the focal rule selected VM 718, and the second run matched all 34 locked population, predictive, and focal-series quantities exactly. Future users should still inspect `locked_summary_comparison.csv`; any discrepancy must be investigated rather than overwritten.
