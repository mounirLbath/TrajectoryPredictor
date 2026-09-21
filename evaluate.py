import sys

import numpy as np
import torch

from av2_data import DATA_ROOT, DT, N_FUT, TYPES, list_scenarios, load_scene, plot_scene
from model import DEVICE, SCALE, FlowModel
from train import load


@torch.no_grad()
def sample(model, feat, type_id, lanes, lane_mask, neighbors, neighbor_mask, neighbor_type, k=6, steps=50):
    scene = [a.repeat_interleave(k, dim=0) for a in (feat, type_id, lanes, lane_mask, neighbors, neighbor_mask, neighbor_type)]
    x = torch.randn(len(feat) * k, N_FUT, 2, device=feat.device)
    for i in range(steps):
        t = torch.full((len(x),), i / steps, device=x.device)
        # each step moves 1/(steps left) of the way toward the predicted clean path
        x = x + (model(x, t, *scene) - x) / (steps - i)
    return (x * SCALE).view(-1, k, N_FUT, 2).cpu().numpy()


def sample_all(model, data, chunk=250):
    scene = data[:1] + data[2:]
    return np.concatenate([sample(model, *(a[i : i + chunk].to(DEVICE) for a in scene)) for i in range(0, len(scene[0]), chunk)])


def metrics(pred, fut):
    error = np.linalg.norm(pred - fut[:, None], axis=-1)
    fde = error[..., -1].min(1)
    return error.mean(-1).min(1).mean(), fde.mean(), (fde > 2).mean()


def constant_velocity(feat):
    seconds = np.arange(1, N_FUT + 1)[None, :, None] * DT
    return (feat[:, -1, 2:4][:, None, :] * seconds)[:, None]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    raw = np.load(DATA_ROOT / "val.npz")
    model = FlowModel().to(DEVICE)
    model.load_state_dict(torch.load("focal_model.pt"))

    samples = sample_all(model, load("val"))
    results = {
        "stay still": np.zeros((len(samples), 1, N_FUT, 2)),
        "constant velocity": constant_velocity(raw["feat"]),
        "flow matching, 1 sample": samples[:, :1],
        "flow matching, 6 samples": samples,
    }
    print(f"{'':26s}{'minADE':>8s}{'minFDE':>8s}{'miss':>8s}")
    for name, pred in results.items():
        ade, fde, miss = metrics(pred, raw["fut"])
        print(f"{name:26s}{ade:8.2f}{fde:8.2f}{miss:8.2f}")

    dirs = list_scenarios("val")
    sideways = np.abs(raw["fut"][:, -1, 1])

    def pick(kind, quantile):
        idx = np.flatnonzero(raw["type"] == TYPES.index(kind))
        return idx[np.argsort(sideways[idx])[int(quantile * (len(idx) - 1))]]

    picks = [pick("vehicle", q) for q in (0.99, 0.95, 0.90)] + [pick(kind, 0.9) for kind in ("pedestrian", "cyclist", "motorcyclist")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, i in zip(axes.ravel(), picks):
        scene = load_scene(dirs[i])
        plot_scene(scene, ax, samples[i])
        reach = 1.4 * max(np.abs(scene["fut"]).max(), 10)
        ax.set_xlim(-0.3 * reach, reach)
        ax.set_ylim(-0.65 * reach, 0.65 * reach)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("chosen by how far the true path curves sideways, not by model score")
    fig.tight_layout()
    fig.savefig("predictions.png", dpi=110)
    if len(sys.argv) > 1:
        fig.savefig(f"results/predictions_{sys.argv[1]}.png", dpi=110)
