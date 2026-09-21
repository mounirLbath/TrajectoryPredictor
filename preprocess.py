"""Cache the focal agent's history features, future and type for each split as arrays."""

from multiprocessing import Pool

import numpy as np

from av2_data import DATA_ROOT, list_scenarios, load_scene


def load(scenario_dir):
    scene = load_scene(scenario_dir)
    return scene["feat"].astype(np.float32), scene["fut"].astype(np.float32), scene["type"]


if __name__ == "__main__":
    for split in ["train", "val"]:
        with Pool() as pool:
            feat, fut, type_ = map(np.array, zip(*pool.map(load, list_scenarios(split), chunksize=64)))
        np.savez(DATA_ROOT / f"{split}.npz", feat=feat, fut=fut, type=type_)
        print(split, feat.shape, fut.shape, type_.shape)
