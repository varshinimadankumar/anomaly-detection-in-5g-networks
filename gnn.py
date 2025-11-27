import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def aggregate(self, x, edge_index):
        src, dst = edge_index
        out = torch.zeros_like(x)

        for s, d in zip(src, dst):
            out[d] += x[s]

        return out

    def forward(self, x, edge_index):
        h = self.aggregate(x, edge_index)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.out(h)
