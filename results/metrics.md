Validation scores per stage, on the 2,000 val scenarios, in meters. minADE / minFDE / miss rate at 2 m.

| Stage | Commit | 1 sample | 6 samples |
|---|---|---|---|
| stay still baseline | | 19.45 / 37.85 / 0.96 | |
| constant velocity baseline | | 4.78 / 12.30 / 0.86 | |
| 1 direct positions | 2290ad0 | 5.05 / 12.45 / 0.97 | 2.74 / 5.64 / 0.81 |
| 2 acceleration and integration | acc70eb | 4.43 / 11.65 / 0.90 | 2.17 / 5.21 / 0.66 |
| 3 lane attention, 10,000 train scenarios | with stage 4 | 3.49 / 8.81 / 0.85 | 1.72 / 3.87 / 0.57 |
| 4 more data, weight averaging, GPU | see git log | 3.40 / 8.69 / 0.84 | 1.51 / 3.30 / 0.52 |

Notes on the images:
- Stage 1 shows six random val scenes, mostly straight vehicles, with jagged samples.
- Stages 2 to 4 show scenes chosen by how far the true path curves sideways, not by model score.
- The checkerboard warm-up images were not saved.

To archive a new stage, run `python evaluate.py <name>` and it also writes `results/predictions_<name>.png`.
