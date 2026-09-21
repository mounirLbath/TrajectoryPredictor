"""Argoverse 2 loader. Scenes are expressed in the focal agent's frame at the last observed step."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).parent / "data" / "av2"
N_HIST, N_FUT = 50, 60


def to_frame(xy, origin, theta):
    c, s = np.cos(theta), np.sin(theta)
    return (xy - origin) @ np.array([[c, -s], [s, c]])


def load_scene(scenario_dir, max_neighbors=32, radius=60.0):
    sid = scenario_dir.name
    df = pd.read_parquet(scenario_dir / f"scenario_{sid}.parquet")
    lane_map = json.load(open(scenario_dir / f"log_map_archive_{sid}.json"))

    focal = df[df.track_id == df.focal_track_id.iloc[0]].sort_values("timestep")
    xy = focal[["position_x", "position_y"]].to_numpy()
    origin, theta = xy[N_HIST - 1], focal.heading.iloc[N_HIST - 1]
    xy = to_frame(xy, origin, theta)

    others = df[(df.track_id != focal.track_id.iloc[0]) & (df.timestep < N_HIST)]
    last = others[others.timestep == N_HIST - 1]
    dist = np.linalg.norm(last[["position_x", "position_y"]].to_numpy() - origin, axis=1)
    keep = last.track_id.to_numpy()[np.argsort(dist)[:max_neighbors]]
    neighbors = np.full((len(keep), N_HIST, 2), np.nan)
    for i, tid in enumerate(keep):
        tr = others[others.track_id == tid]
        neighbors[i, tr.timestep] = to_frame(tr[["position_x", "position_y"]].to_numpy(), origin, theta)

    lanes = [np.array([[p["x"], p["y"]] for p in seg["centerline"]]) for seg in lane_map["lane_segments"].values()]
    lanes = [to_frame(l, origin, theta) for l in lanes if np.linalg.norm(l - origin, axis=1).min() < radius]

    return dict(id=sid, hist=xy[:N_HIST], fut=xy[N_HIST:], neighbors=neighbors, lanes=lanes)


def list_scenarios(split="val"):
    return sorted(p for p in (DATA_ROOT / split).iterdir() if p.is_dir())


def plot_scene(scene, ax, preds=None):
    for lane in scene["lanes"]:
        ax.plot(*lane.T, color="0.8", lw=1)
    for nb in scene["neighbors"]:
        ax.plot(*nb.T, color="tab:blue", lw=1, alpha=0.6)
    for p in ([] if preds is None else preds):
        ax.plot(*p.T, color="tab:orange", lw=1.2, alpha=0.7)
    ax.plot(*scene["hist"].T, color="k", lw=2, label="history")
    ax.plot(*scene["fut"].T, color="tab:green", lw=2, label="future")
    ax.set_aspect("equal")
    ax.set_xlim(-40, 80)
    ax.set_ylim(-60, 60)
    ax.set_title(scene["id"][:8], fontsize=9)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dirs = list_scenarios("val")
    picks = np.random.default_rng(0).choice(len(dirs), 6, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, i in zip(axes.ravel(), picks):
        plot_scene(load_scene(dirs[i]), ax)
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("scenes.png", dpi=110)
    print(f"{len(dirs)} scenarios, saved scenes.png")
