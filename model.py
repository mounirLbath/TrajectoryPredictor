import math

import torch
import torch.nn as nn

from av2_data import DT, N_FUT, TYPES

SCALE = 20.0 # meters, brings the future to a spread of about 1
N_FEAT = 6
A_MAX = 0.7 * 9.81 # tire friction limit, 0.7 g as in the paper
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def integrate(u, v0):
    norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    acceleration = A_MAX * torch.tanh(norm) * u / norm
    velocity = v0[:, None] + acceleration.cumsum(1) * DT
    return velocity.cumsum(1) * DT / SCALE


class FlowModel(nn.Module):
    def __init__(self, hidden=256, type_dim=16, time_dim=32, width=512, lane_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.type_embedding = nn.Embedding(len(TYPES), type_dim)
        self.history_encoder = nn.GRU(N_FEAT, hidden, batch_first=True)
        self.lane_points = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64))
        self.lane_token = nn.Linear(64, lane_dim)
        self.null_lane = nn.Parameter(torch.zeros(1, 1, lane_dim))
        query_dim = 2 * N_FUT + hidden + type_dim + time_dim
        self.query = nn.Linear(query_dim, lane_dim)
        self.lane_attention = nn.MultiheadAttention(lane_dim, 4, batch_first=True)
        self.acceleration_net = nn.Sequential(
            nn.Linear(query_dim + lane_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2 * N_FUT),
        )

    def time_embedding(self, t):
        half = self.time_dim // 2
        freqs = 1000 * torch.exp(-math.log(1000) * torch.arange(half, device=t.device) / half)
        angles = t[:, None] * freqs
        return torch.cat([angles.sin(), angles.cos()], dim=1)

    def forward(self, x, t, history, type_id, lanes, lane_mask):
        history_code = self.history_encoder(history)[1][-1]
        condition = torch.cat([history_code, self.type_embedding(type_id)], dim=1)
        inputs = torch.cat([x.flatten(1), condition, self.time_embedding(t)], dim=1)
        # the null token keeps attention defined when a scene has no lanes at all
        tokens = torch.cat([self.null_lane.expand(len(x), -1, -1), self.lane_token(self.lane_points(lanes).max(2).values)], dim=1)
        keep = torch.cat([torch.ones_like(lane_mask[:, :1]), lane_mask], dim=1)
        road = self.lane_attention(self.query(inputs)[:, None], tokens, tokens, key_padding_mask=~keep)[0][:, 0]
        u = self.acceleration_net(torch.cat([inputs, road], dim=1)).view(-1, N_FUT, 2)
        return integrate(u, history[:, -1, 2:4] * SCALE)


def normalise(feat, fut=None):
    feat = feat.clone()
    feat[..., :4] /= SCALE
    return feat if fut is None else (feat, fut / SCALE)
