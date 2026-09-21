import numpy as np
import torch

from av2_data import DATA_ROOT, N_FUT, list_scenarios, load_scene, plot_scene
from model import SCALE, FlowModel
from train import load

DT = 0.1


@torch.no_grad()
def sample(model, feat, type_id, k=6, steps=50):
    feat, type_id = feat.repeat_interleave(k, dim=0), type_id.repeat_interleave(k)
    x = torch.randn(len(feat), N_FUT, 2)
    for i in range(steps):
        t = torch.full((len(x),), i / steps)
        x = x + model(x, t, feat, type_id) / steps
    return (x * SCALE).view(-1, k, N_FUT, 2).numpy()


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
    feat, _, type_id = load("val")
    model = FlowModel()
    model.load_state_dict(torch.load("focal_model.pt"))

    samples = np.concatenate([sample(model, feat[i : i + 500], type_id[i : i + 500]) for i in range(0, len(feat), 500)])
    results = {
        "stay still": np.zeros((len(feat), 1, N_FUT, 2)),
        "constant velocity": constant_velocity(raw["feat"]),
        "flow matching, 1 sample": samples[:, :1],
        "flow matching, 6 samples": samples,
    }
    print(f"{'':26s}{'minADE':>8s}{'minFDE':>8s}{'miss':>8s}")
    for name, pred in results.items():
        ade, fde, miss = metrics(pred, raw["fut"])
        print(f"{name:26s}{ade:8.2f}{fde:8.2f}{miss:8.2f}")

    dirs = list_scenarios("val")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, i in zip(axes.ravel(), np.random.default_rng(0).choice(len(dirs), 6, replace=False)):
        plot_scene(load_scene(dirs[i]), ax, samples[i])
    fig.tight_layout()
    fig.savefig("predictions.png", dpi=110)
