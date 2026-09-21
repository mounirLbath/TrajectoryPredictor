import numpy as np
import torch

from av2_data import DATA_ROOT
from model import DEVICE, SCALE, FlowModel, normalise


def load(split):
    data = np.load(DATA_ROOT / f"{split}.npz")
    feat, fut = normalise(torch.tensor(data["feat"]), torch.tensor(data["fut"]))
    lanes = torch.tensor(data["lanes"])
    lanes[..., :2] /= SCALE
    return feat, fut, torch.tensor(data["type"]), lanes, torch.tensor(data["lane_mask"])


def flow_loss(model, feat, fut, type_id, lanes, lane_mask, generator=None):
    x0 = torch.randn(fut.shape, generator=generator).to(fut.device)
    t = torch.rand(len(fut), generator=generator).to(fut.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * fut
    return ((model(xt, t, feat, type_id, lanes, lane_mask) - fut) ** 2).mean()


def val_loss(model, val):
    # fixed noise and times so the loss is comparable across checkpoints
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        return np.mean([flow_loss(model, *(a[i : i + 500] for a in val), generator=generator).item() for i in range(0, len(val[1]), 500)])


if __name__ == "__main__":
    torch.manual_seed(0)
    train, val = ([a.to(DEVICE) for a in load(split)] for split in ("train", "val"))
    model = FlowModel().to(DEVICE)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    steps, batch = 20_000, 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    losses = []
    for step in range(1, steps + 1):
        idx = torch.randint(len(train[1]), (batch,), device=DEVICE)
        loss = flow_loss(model, *(a[idx] for a in train))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update_parameters(model)
        losses.append(loss.item())
        if step % 500 == 0:
            print(f"step {step:6d}  train {np.mean(losses[-500:]):.3f}  val {val_loss(model, val):.3f}  val ema {val_loss(ema.module, val):.3f}", flush=True)
    torch.save(ema.module.cpu().state_dict(), "focal_model.pt")
