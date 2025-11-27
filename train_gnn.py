import pandas as pd
import torch
from torch import nn, optim
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.preprocessing.feature_engineering import build_feature_matrix
from src.models.gnn import GraphSAGE


df = pd.read_parquet("data/processed/labeled_with_metadata.parquet")

X = build_feature_matrix(df).values
X = torch.tensor(X, dtype=torch.float32)

labels = torch.tensor(df["label"].values, dtype=torch.long)

# build small graph (src → dst)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

nodes = pd.concat([df["source"], df["destination"]]).astype(str)
le.fit(nodes)

src_ids = le.transform(df["source"].astype(str))
dst_ids = le.transform(df["destination"].astype(str))

edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)

in_dim = X.shape[1]
print("Training GNN with input dim =", in_dim)

model = GraphSAGE(in_dim=in_dim, hidden_dim=64, out_dim=2)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(20):
    opt.zero_grad()
    out = model(X, edge_index)
    loss = crit(out, labels)
    loss.backward()
    opt.step()
    print(epoch, loss.item())

torch.save(model.state_dict(), "models/gnn.pt")
print("✓ GNN saved")
