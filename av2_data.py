"""Argoverse 2 loader, scenes are expressed in the focal agent's frame at the last observed step"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).parent / "data" / "av2"
N_HIST, N_FUT = 50, 60 # number of historical timesteps/future timesteps that we are predicting
DT = 0.1
N_LANES, N_LANE_POINTS = 128, 10
N_NEIGHBORS, N_NEIGHBOR_STEPS = 32, 20
TYPES = ["vehicle", "pedestrian", "motorcyclist", "cyclist", "bus", "static", "background", "construction", "riderless_bicycle", "unknown"]


def to_frame(xy, origin, theta):
    c, s = np.cos(theta), np.sin(theta)
    return (xy - origin) @ np.array([[c, -s], [s, c]])


def load_scene(scenario_dir, max_neighbors=32, radius=100.0):
    sid = scenario_dir.name
    df = pd.read_parquet(scenario_dir / f"scenario_{sid}.parquet")
    lane_map = json.load(open(scenario_dir / f"log_map_archive_{sid}.json"))

    focal = df[df.track_id == df.focal_track_id.iloc[0]].sort_values("timestep")
    assert len(focal) == N_HIST + N_FUT
    xy = focal[["position_x", "position_y"]].to_numpy()
    # "now" is the last observed step, everything is expressed relative to it
    origin, theta = xy[N_HIST - 1], focal.heading.iloc[N_HIST - 1]
    xy = to_frame(xy, origin, theta)
    velocity = to_frame(focal[["velocity_x", "velocity_y"]].to_numpy(), 0, theta)
    heading = focal.heading.to_numpy() - theta
    feat = np.column_stack([xy, velocity, np.cos(heading), np.sin(heading)])[:N_HIST]

    others = df[(df.track_id != focal.track_id.iloc[0]) & (df.timestep < N_HIST)]
    last = others[others.timestep == N_HIST - 1]
    dist = np.linalg.norm(last[["position_x", "position_y"]].to_numpy() - origin, axis=1)
    keep = last.track_id.to_numpy()[np.argsort(dist)[:max_neighbors]]
    # NaN where a neighbour was not observed, zeros would look like a car at the origin
    neighbors = np.full((len(keep), N_HIST, 2), np.nan)
    neighbor_types = []
    for i, tid in enumerate(keep):
        tr = others[others.track_id == tid]
        neighbors[i, tr.timestep] = to_frame(tr[["position_x", "position_y"]].to_numpy(), origin, theta)
        neighbor_types.append(TYPES.index(tr.object_type.iloc[0]))

    segments = lane_map["lane_segments"].values()
    lanes = {seg["id"]: to_frame(np.array([[p["x"], p["y"]] for p in seg["centerline"]]), origin, theta) for seg in segments}
    lanes = {i: l for i, l in lanes.items() if np.linalg.norm(l, axis=1).min() < radius}
    links = {seg["id"]: [seg["successors"], seg["predecessors"], [seg["left_neighbor_id"]], [seg["right_neighbor_id"]]] for seg in segments if seg["id"] in lanes}

    return dict(id=sid, feat=feat, type=TYPES.index(focal.object_type.iloc[0]), hist=xy[:N_HIST], fut=xy[N_HIST:], neighbors=neighbors, neighbor_types=neighbor_types, lanes=list(lanes.values()), lane_ids=list(lanes), lane_links=links)


def lane_tokens(lanes, ids, links):
    order = np.argsort([np.linalg.norm(l, axis=1).min() for l in lanes])[:N_LANES]
    slot = {ids[j]: i for i, j in enumerate(order)}
    tokens = np.zeros((N_LANES, N_LANE_POINTS, 4), dtype=np.float32)
    # one adjacency per edge type: successor, predecessor, left neighbour, right neighbour
    edges = np.zeros((4, N_LANES, N_LANES), dtype=bool)
    for i, j in enumerate(order):
        length = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(lanes[j], axis=0), axis=1))])
        along = np.linspace(0, length[-1], N_LANE_POINTS)
        points = np.column_stack([np.interp(along, length, lanes[j][:, k]) for k in range(2)])
        direction = np.gradient(points, axis=0)
        tokens[i] = np.column_stack([points, direction / np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-6)])
        for kind, targets in enumerate(links[ids[j]]):
            for target in targets:
                if target in slot:
                    edges[kind, i, slot[target]] = True
    return tokens, np.arange(N_LANES) < len(lanes), edges


def neighbor_tokens(neighbors, types):
    recent = neighbors[:, -N_NEIGHBOR_STEPS:]
    tokens = np.zeros((N_NEIGHBORS, N_NEIGHBOR_STEPS, 3), dtype=np.float32)
    tokens[: len(recent), :, :2] = np.nan_to_num(recent)
    tokens[: len(recent), :, 2] = ~np.isnan(recent[..., 0])
    return tokens, np.arange(N_NEIGHBORS) < len(recent), np.pad(np.array(types, dtype=int), (0, N_NEIGHBORS - len(types)))


SUBSET = {"train": 30_000, "val": 2_000}


def list_scenarios(split="val"):
    return sorted(p for p in (DATA_ROOT / split).iterdir() if p.is_dir())[: SUBSET[split]]


def plot_scene(scene, ax, preds=None):
    for lane in scene["lanes"]:
        ax.plot(*lane.T, color="0.8", lw=1)
    for nb in scene["neighbors"]:
        ax.plot(*nb.T, color="tab:blue", lw=1, alpha=0.6)
    for i, p in enumerate([] if preds is None else preds):
        ax.plot(*p.T, color="tab:orange", lw=1.2, alpha=0.7, label="samples" if i == 0 else None)
    ax.plot(*scene["hist"].T, color="k", lw=2, label="history")
    ax.plot(*scene["fut"].T, color="tab:green", lw=2, label="future")
    ax.annotate(TYPES[scene["type"]], (0, 0), xytext=(6, 8), textcoords="offset points", weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-40, 80)
    ax.set_ylim(-60, 60)
    ax.set_title(scene["id"][:8], fontsize=9)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dirs = list_scenarios("val")
    scenes = {}
    for d in dirs:
        scene = load_scene(d)
        scenes.setdefault(scene["type"], scene)
        if len(scenes) == 6:
            break
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, scene in zip(axes.ravel(), scenes.values()):
        plot_scene(scene, ax)
    for ax in axes.ravel()[len(scenes):]:
        ax.axis("off")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("scenes.png", dpi=110)
    print(f"{len(dirs)} scenarios, saved scenes.png")
