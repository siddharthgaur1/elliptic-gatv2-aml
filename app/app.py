"""Streamlit demo: inspect GATv2's call on individual Elliptic test transactions.

Loads the committed trained models from results/models/ -- does not retrain.
Run with: streamlit run app/app.py
"""
import sys
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ILLICIT, LICIT, UNKNOWN, load_data
from src.evaluate import load_trained_nn, load_trained_rf
from src.explain import get_attention_for_node

st.set_page_config(page_title="Elliptic GATv2 AML Demo", layout="wide")

LABEL_NAME = {LICIT: "licit", ILLICIT: "illicit", UNKNOWN: "unknown"}


@st.cache_resource
def load_everything():
    _, data = load_data()
    gatv2 = load_trained_nn("gatv2", data)
    rf = load_trained_rf()
    return data, gatv2, rf


data, gatv2_model, rf_model = load_everything()
test_idx = data.test_mask.nonzero(as_tuple=True)[0].tolist()

st.title("Elliptic GATv2 AML Detector -- Test Set Explorer")
st.caption(
    "Pick a test-set Bitcoin transaction node, see its 1-hop neighborhood, "
    "GATv2's attention over that neighborhood, and how GATv2's call compares to Random Forest's."
)

with st.sidebar:
    st.header("Pick a transaction")
    illicit_only = st.checkbox("Show only ground-truth illicit nodes", value=True)
    candidates = [i for i in test_idx if (not illicit_only or data.y[i].item() == ILLICIT)]
    node_id = st.selectbox("Test node index", candidates, index=0)
    show_rf = st.checkbox("Compare with Random Forest", value=True)

# ---- predictions ----
with torch.no_grad():
    gatv2_logits = gatv2_model(data.x, data.edge_index)
    gatv2_probs = torch.softmax(gatv2_logits, dim=1)
gatv2_pred = int(gatv2_probs[node_id].argmax())
gatv2_conf = float(gatv2_probs[node_id, gatv2_pred])

rf_probs = rf_model.predict_proba(data.x[node_id].unsqueeze(0).numpy())[0]
rf_pred = int(rf_probs.argmax())
rf_conf = float(rf_probs[rf_pred])

true_label = int(data.y[node_id])

col1, col2, col3 = st.columns(3)
col1.metric("Ground truth", LABEL_NAME[true_label])
col2.metric("GATv2 prediction", LABEL_NAME[gatv2_pred], f"{gatv2_conf:.1%} confidence")
if show_rf:
    col3.metric("Random Forest prediction", LABEL_NAME[rf_pred], f"{rf_conf:.1%} confidence")

if show_rf and gatv2_pred != rf_pred:
    st.warning("GATv2 and Random Forest disagree on this node.")

# ---- k-hop subgraph + attention ----
st.subheader("1-hop neighborhood with GATv2 attention")
subset, center_local, ei_local, alpha = get_attention_for_node(gatv2_model, data, node_id, num_hops=1)

g = nx.DiGraph()
for i, n in enumerate(subset.tolist()):
    g.add_node(i, global_id=n, label=LABEL_NAME[int(data.y[n])])
for e in range(ei_local.shape[1]):
    src, dst = int(ei_local[0, e]), int(ei_local[1, e])
    g.add_edge(src, dst, weight=float(alpha[e]))

if len(g.nodes) == 0:
    st.info("No 1-hop neighbors found for this node (it may be isolated in the graph).")
else:
    pos = nx.spring_layout(g, seed=0)
    edge_traces = []
    for u, v, d in g.edges(data=True):
        w = d["weight"]
        edge_traces.append(
            go.Scatter(
                x=[pos[u][0], pos[v][0], None],
                y=[pos[u][1], pos[v][1], None],
                mode="lines",
                line=dict(width=1 + 6 * w, color=f"rgba(192,57,43,{0.2 + 0.8 * w})"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    node_x, node_y, node_color, node_text = [], [], [], []
    color_map = {"licit": "#2980b9", "illicit": "#c0392b", "unknown": "#95a5a6"}
    for n, d in g.nodes(data=True):
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        is_center = d["global_id"] == node_id
        node_color.append("#f1c40f" if is_center else color_map[d["label"]])
        node_text.append(f"node {d['global_id']} ({d['label']}){' -- SELECTED' if is_center else ''}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=16, color=node_color, line=dict(width=1, color="black")),
        text=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Edge thickness/opacity = GATv2 attention weight (mean over heads, last layer). "
        "Yellow = selected node. Blue = licit, red = illicit, gray = unknown."
    )
