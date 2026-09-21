"""Cache the focal agent's history features, future, type, lane tokens and neighbour tokens for each split as arrays"""

from multiprocessing import Pool

import numpy as np

from av2_data import DATA_ROOT, lane_tokens, list_scenarios, load_scene, neighbor_tokens


def load(scenario_dir):
    scene = load_scene(scenario_dir)
    lanes, lane_mask = lane_tokens(scene["lanes"])
    neighbors, neighbor_mask, neighbor_type = neighbor_tokens(scene["neighbors"], scene["neighbor_types"])
    return scene["feat"].astype(np.float32), scene["fut"].astype(np.float32), scene["type"], lanes, lane_mask, neighbors, neighbor_mask, neighbor_type


if __name__ == "__main__":
    for split in ["train", "val"]:
        with Pool() as pool:
            arrays = map(np.array, zip(*pool.map(load, list_scenarios(split), chunksize=64)))
        names = ["feat", "fut", "type", "lanes", "lane_mask", "neighbors", "neighbor_mask", "neighbor_type"]
        np.savez(DATA_ROOT / f"{split}.npz", **dict(zip(names, arrays)))
        print(split, "saved", names)
