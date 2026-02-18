# Spa: Model Comparison

This page explains `examples/spa/spa_model_comparison.py`.

## Learning goal

Quantify the impact of model complexity in a reproducible way.

## Comparison workflow in the script

1. Run single-track model on Spa.
2. Calibrate point-mass friction to single-track lateral envelope.
3. Run calibrated point-mass model.
4. Export KPI deltas and speed-trace overlay.

## Why calibration matters

Without calibration, comparison can be biased by arbitrary friction settings.
Calibration aligns lateral capability first, so differences are easier to
attribute to model structure rather than parameter mismatch.

## Main artifacts

1. `examples/output/spa/comparison/comparison_kpis.json`
2. `examples/output/spa/comparison/speed_trace_comparison.png`
3. `examples/output/spa/comparison/single_track/kpis.json`
4. `examples/output/spa/comparison/point_mass_calibrated/kpis.json`

## How to read delta metrics

- `lap_time_delta`: single-track minus point-mass lap time.
- `mean_abs_speed_delta`: average absolute speed-gap over the full lap.
- `max_abs_speed_delta`: largest local disagreement.

Interpretation rule:
Use local speed differences along arc length to explain global KPI deltas.

## Student-friendly analysis checklist

1. Identify top 3 sectors with largest speed difference.
2. Check whether those sectors are braking-limited or cornering-limited.
3. Decide whether point-mass is sufficient for your study question.
4. Write one sentence for each model’s strongest use case.

## Suggested exercise

Repeat the comparison with:

1. reduced downforce (`lift_coefficient` lower),
2. unchanged tire parameters,
3. same runtime config.

Then document whether complexity benefits increase or decrease.
