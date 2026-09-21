import numpy as np
import torch

from av2_data import DATA_ROOT
from model import FlowModel, normalise


def load(split):
    data = np.load(DATA_ROOT / f"{split}.npz")
    feat, fut = normalise(torch.tensor(data["feat"]), torch.tensor(data["fut"]))
    return feat, fut, torch.tensor(data["type"])


def flow_loss(model, feat, fut, type_id, generator=None):
    x0 = torch.randn(fut.shape, generator=generator)
    t = torch.rand(len(fut), generator=generator)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * fut
    return ((model(xt, t, feat, type_id) - (fut - x0)) ** 2).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    train, val = load("train"), load("val")
    model = FlowModel()
    steps, batch = 10_000, 256
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    losses = []
    for step in range(1, steps + 1):
        idx = torch.randint(len(train[1]), (batch,))
        loss = flow_loss(model, *(a[idx] for a in train))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step % 500 == 0:
            # fixed noise and times on val so the loss is comparable across checkpoints
            with torch.no_grad():
                val_loss = flow_loss(model, *val, generator=torch.Generator().manual_seed(0))
            print(f"step {step:6d}  train {np.mean(losses[-500:]):.3f}  val {val_loss.item():.3f}", flush=True)
    torch.save(model.state_dict(), "focal_model.pt")
