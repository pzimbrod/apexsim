# Track Data Provenance

## Spa-Francorchamps

- Local file: `data/spa_francorchamps.csv`
- Source (centerline + widths):
  - `https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/Spa.csv`
  - Repository: `https://github.com/TUMFTM/racetrack-database`
- Retrieval date: 2026-02-15

The imported source provides:
- `x_m`, `y_m` (smoothed centerline)
- `w_tr_right_m`, `w_tr_left_m` (track widths)

Our simulation format requires `x`, `y`, `elevation`, `banking`.
Because elevation and banking are not supplied in the source file, this dataset currently uses:
- `elevation = 0.0`
- `banking = 0.0`

You can reproduce/update the import with:

```bash
source .venv/bin/activate
python scripts/import_spa_from_tumftm.py
```
