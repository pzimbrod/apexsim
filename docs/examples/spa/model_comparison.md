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

## Theoretical calibration note

The calibration step fits an effective point-mass friction value so that
point-mass lateral limits approximate single-track lateral limits over a speed set.
Conceptually, this is a least-squares fit:

$$
\mu^\star
= \arg\min_{\mu}\sum_i\left(\mu\,a_n(v_i)-a_{y,\mathrm{lim},\mathrm{single\text{-}track}}(v_i)\right)^2
$$

This keeps cross-model comparison focused on structure, not arbitrary parameter
offsets.

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

## Neutral comparison protocol

1. Keep track data and runtime config identical for both models.
2. Compare lap time, mean absolute speed delta, and local peak delta together.
3. Attribute observed differences to specific mechanisms:
   yaw/axle dynamics, tire-model detail, or simplified envelopes.
4. State clearly where point-mass is sufficient and where single-track is required.
