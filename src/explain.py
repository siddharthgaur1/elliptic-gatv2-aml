"""Attention inspection: for a few correctly-flagged illicit test nodes,
pull GATv2's last-layer attention weights over their 1-hop neighborhood and
plot which neighbors were weighted most.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph

from src.data import ILLICIT, load_data
from src.evaluate import load_trained_nn


def get_attention_for_node(model, data, node_idx, num_hops=1):
    """Return (subgraph_nodes, center_local_idx, edge_index_local, alpha) for the
    LAST GATv2 layer's attention over node_idx's k-hop neighborhood."""
    subset, edge_index_local, mapping, _edge_mask = k_hop_subgraph(
        node_idx, num_hops, data.edge_index, relabel_nodes=True
    )
    x_sub = data.x[subset]
    model.eval()
    with torch.no_grad():
        _, attn = model(x_sub, edge_index_local, return_attention_weights=True)
    last_ei, last_alpha = attn[-1]  # (2, E'), (E', heads)
    alpha_mean = last_alpha.mean(dim=1)  # average over heads
    return subset, mapping.item(), last_ei, alpha_mean


def top_neighbor_weights(subset, center_local, edge_index_local, alpha, top_k=8):
    """Edges pointing INTO the center node (GATv2Conv aggregates dst<-src)."""
    mask = edge_index_local[1] == center_local
    src_nodes = edge_index_local[0][mask]
    weights = alpha[mask]
    order = torch.argsort(weights, descending=True)[:top_k]
    global_ids = subset[src_nodes[order]]
    return global_ids.tolist(), weights[order].tolist()


def plot_attention(node_global_id, neighbor_ids, neighbor_weights, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"node {n}" for n in neighbor_ids]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, neighbor_weights, color="#c0392b")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("GATv2 attention weight (mean over heads)")
    ax.set_title(f"Top neighbors by attention -- illicit tx node {node_global_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(out_dir="results", n_examples=3):
    _, data = load_data()
    model = load_trained_nn("gatv2", data, out_dir=out_dir)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

    test_mask = data.test_mask
    correct_illicit = ((pred == ILLICIT) & (data.y == ILLICIT) & test_mask).nonzero(as_tuple=True)[0]
    print(f"correctly-flagged illicit test nodes: {len(correct_illicit)}")

    out_dir_p = Path(out_dir) / "figures"
    out_dir_p.mkdir(parents=True, exist_ok=True)

    chosen = correct_illicit[:n_examples].tolist()
    for node_idx in chosen:
        subset, center_local, ei_local, alpha = get_attention_for_node(model, data, node_idx)
        neighbor_ids, weights = top_neighbor_weights(subset, center_local, ei_local, alpha)
        if not neighbor_ids:
            print(f"node {node_idx}: no incoming neighbors in 1-hop subgraph, skipping")
            continue
        out_path = out_dir_p / f"attention_node_{node_idx}.png"
        plot_attention(node_idx, neighbor_ids, weights, out_path)
        print(f"node {node_idx}: saved {out_path}")


if __name__ == "__main__":
    main()
