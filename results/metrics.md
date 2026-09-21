Validation scores per stage, on the 2,000 val scenarios, in meters. minADE / minFDE / miss rate at 2 m.

| Stage | Commit | 1 sample | 6 samples |
|---|---|---|---|
| stay still baseline | | 19.45 / 37.85 / 0.96 | |
| constant velocity baseline | | 4.78 / 12.30 / 0.86 | |
| 1 direct positions | 2290ad0 | 5.05 / 12.45 / 0.97 | 2.74 / 5.64 / 0.81 |
| 2 acceleration and integration | acc70eb | 4.43 / 11.65 / 0.90 | 2.17 / 5.21 / 0.66 |
| 3 lane attention, 10,000 train scenarios | 94cbfcb | 3.49 / 8.81 / 0.85 | 1.72 / 3.87 / 0.57 |
| 4 more data, weight averaging, GPU | 94cbfcb | 3.40 / 8.69 / 0.84 | 1.51 / 3.30 / 0.52 |
| 5 neighbour attention | 88562b4 | 3.15 / 8.00 / 0.85 | 1.49 / 3.30 / 0.52 |
| 6 lane GNN, 2 rounds of message passing | see git log | 2.97 / 7.49 / 0.83 | 1.91 / 4.46 / 0.54 |

Notes on the images:
- Stage 1 shows six random val scenes, mostly straight vehicles, with jagged samples.
- Stages 2 to 6 show scenes chosen by how far the true path curves sideways, not by model score.
- The checkerboard warm-up images were not saved.

To archive a new stage, run `python evaluate.py <name>` and it also writes `results/predictions_<name>.png`.

Stage 5 by how much the true path curves sideways, 6 samples, minADE / minFDE:

| Path type | Scenes | Constant velocity | Stage 5 |
|---|---|---|---|
| straight, under 2 m | 1320 | 3.91 / 10.19 | 1.22 / 2.58 |
| gentle, 2 to 8 m | 387 | 4.27 / 10.73 | 1.72 / 3.98 |
| sharp, 8 m or more | 293 | 9.34 / 23.90 | 2.42 / 5.68 |

Stage 5 reliance test, 6 samples, all scenes, minADE / minFDE:

| Context given to the trained model | Score |
|---|---|
| its own lanes and neighbours | 1.49 / 3.30 |
| another scene's lanes | 3.59 / 9.27 |
| lanes hidden | 2.52 / 6.13 |
| another scene's neighbours | 2.07 / 4.83 |
| neighbours hidden | 1.70 / 3.88 |

One run per stage with one seed. The stage 5 weights are kept locally in `results/focal_model_5_neighbors.pt`, which git ignores.

Stage 6 overfits: train loss 0.003 against val 0.015, where stage 5 had 0.007 against 0.009. The single sample improved but the six samples collapsed onto nearly the same path, so best-of-6 got worse. Removing the edges at test time raises minADE from 1.91 to 3.63, so the graph is used, it just memorises.

Stage 6 by curve type, 6 samples, minADE / minFDE, stage 5 in brackets:

| Path type | Stage 6 | Stage 5 |
|---|---|---|
| straight | 1.59 / 3.62 | 1.22 / 2.58 |
| gentle | 2.05 / 4.76 | 1.72 / 3.98 |
| sharp | 3.17 / 7.85 | 2.42 / 5.68 |

Weights: `results/focal_model_5_neighbors.pt` loads into `FlowModel(gnn_layers=0)`, `results/focal_model_6_lane_gnn.pt` into the default model. Both are gitignored.
