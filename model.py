import math

import torch
import torch.nn as nn

from av2_data import DT, N_FUT, TYPES

SCALE = 20.0 # meters, brings the future to a spread of about 1
N_FEAT = 6
A_MAX = 0.7 * 9.81 # tire friction limit, 0.7 g as in the paper


def integrate(u, v0):
    norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    acceleration = A_MAX * torch.tanh(norm) * u / norm
    velocity = v0[:, None] + acceleration.cumsum(1) * DT
    return velocity.cumsum(1) * DT / SCALE


class FlowModel(nn.Module):
    def __init__(self, hidden=256, type_dim=16, time_dim=32, width=512):
        super().__init__()
        self.time_dim = time_dim
        self.type_embedding = nn.Embedding(len(TYPES), type_dim)
        self.history_encoder = nn.GRU(N_FEAT, hidden, batch_first=True)
        self.acceleration_net = nn.Sequential(
            nn.Linear(2 * N_FUT + hidden + type_dim + time_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2 * N_FUT),
        )

    def time_embedding(self, t):
        half = self.time_dim // 2
        freqs = 1000 * torch.exp(-math.log(1000) * torch.arange(half) / half)
        angles = t[:, None] * freqs
        return torch.cat([angles.sin(), angles.cos()], dim=1)

    def forward(self, x, t, history, type_id):
        history_code = self.history_encoder(history)[1][-1]
        condition = torch.cat([history_code, self.type_embedding(type_id)], dim=1)
        inputs = torch.cat([x.flatten(1), condition, self.time_embedding(t)], dim=1)
        u = self.acceleration_net(inputs).view(-1, N_FUT, 2)
        return integrate(u, history[:, -1, 2:4] * SCALE)


def normalise(feat, fut=None):
    feat = feat.clone()
    feat[..., :4] /= SCALE
    return feat if fut is None else (feat, fut / SCALE)
