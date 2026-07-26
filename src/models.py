"""GATv2 / GCN / MLP node classifiers sharing the same 2-class output head."""
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv


class GATv2Net(nn.Module):
    def __init__(self, in_channels, hidden=64, heads=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden * heads, hidden, heads=heads, dropout=dropout))
        last_in = hidden * heads if num_layers > 1 else hidden
        self.convs.append(GATv2Conv(last_in, hidden, heads=1, concat=False, dropout=dropout))
        self.out = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, return_attention_weights=False):
        attn = []
        for i, conv in enumerate(self.convs):
            if return_attention_weights:
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                attn.append((ei, alpha))
            else:
                x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.out(x)
        if return_attention_weights:
            return logits, attn
        return logits


class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.out = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


class MLPNet(nn.Module):
    """Ignores edge_index entirely -- isolates signal in node features alone."""

    def __init__(self, in_channels, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, edge_index=None, **kwargs):
        return self.net(x)


def build_model(name, in_channels, **kwargs):
    name = name.lower()
    if name == "gatv2":
        return GATv2Net(in_channels, **kwargs)
    if name == "gcn":
        return GCNNet(in_channels, **kwargs)
    if name == "mlp":
        return MLPNet(in_channels, **kwargs)
    raise ValueError(f"unknown model {name}")
