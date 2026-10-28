"""Tab 5 (graph / network): a feature-correlation network and a class "confusability" network."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.feature_selection import f_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from common import CLASS_COLORS, CLASS_LABELS, CLASSES, alt_text, feature_names, load_tables, maybe_table, style
from noisedx.config import CLASS_SYMBOLS

COMMUNITY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
OTHER = "#8a8984"


@st.cache_data
def _table() -> pd.DataFrame:
    meta, feats = load_tables()
    return meta.merge(feats, on="id")


@st.cache_data
def _feature_graph(strategy: str, threshold: float) -> tuple[nx.Graph, dict, pd.DataFrame]:
    df = _table()
    sub = df[df.strategy == strategy]
    names = feature_names()
    corr = sub[names].corr().to_numpy()
    F, _ = f_classif(sub[names], sub.label.astype(str))
    G = nx.Graph()
    for i, n in enumerate(names):
        G.add_node(n, F=float(F[i]))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) >= threshold:
                G.add_edge(names[i], names[j], r=float(corr[i, j]), weight=abs(corr[i, j]))
    comms = sorted(greedy_modularity_communities(G, weight="weight"), key=len, reverse=True)
    comm_of = {n: k for k, c in enumerate(comms) for n in c}
    # Circular layout ordered by community, so each group sits together as an arc and every node can be labelled.
    order = sorted(names, key=lambda n: (len(comms[comm_of[n]]) == 1, comm_of[n], -G.nodes[n]["F"]))
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(order), endpoint=False)
    pos = {n: np.array([np.cos(t), np.sin(t)]) for n, t in zip(order, ang)}
    deg = nx.degree_centrality(G)
    nodes = pd.DataFrame({
        "feature": names, "community": [comm_of[n] + 1 for n in names],
        "connections": [G.degree(n) for n in names], "degree_centrality": [round(deg[n], 3) for n in names],
        "class_separation_F": np.round(F, 1),
    }).sort_values(["community", "class_separation_F"], ascending=[True, False])
    return G, pos, nodes


@st.cache_data
def _class_graph(strategy: str, k: int) -> pd.DataFrame:
    """Share of each class's k nearest neighbours (38-D standardised features) that belong to another class."""
    df = _table()
    sub = df[df.strategy == strategy].reset_index(drop=True)
    X = StandardScaler().fit_transform(sub[feature_names()])
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    lab = sub.label.astype(str).to_numpy()
    rows = []
    for a in CLASSES:
        neigh = lab[idx[lab == a, 1:]].ravel()
        for b in CLASSES:
            rows.append({"from": a, "to": b, "share": float(np.mean(neigh == b))})
    m = pd.DataFrame(rows).pivot(index="from", columns="to", values="share").loc[CLASSES, CLASSES]
    return m


def render() -> None:
    st.markdown("**Question:** the 38 features are not independent. Which ones are measuring the same thing, and which "
                "ones actually separate the classes?")
    c1, c2 = st.columns(2)
    strategy = c1.radio("Strategy", ["narrow", "diverse"], index=1, horizontal=True, key="nw_strategy",
                        format_func=str.title)
    thr = c2.slider("Draw a link when |correlation| is at least", 0.3, 0.9, 0.5, 0.05, key="nw_thr")
    G, pos, nodes = _feature_graph(strategy, thr)

    fig = go.Figure()
    for sign, colour, dash, label in [(1, "#8a8984", "solid", "Positive correlation"),
                                      (-1, "#e34948", "dot", "Negative correlation")]:
        xs, ys = [], []
        for u, v, d in G.edges(data=True):
            if np.sign(d["r"]) == sign:
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=colour, width=1.5, dash=dash), name=label,
                        hoverinfo="skip", opacity=0.7)
    F = nodes.set_index("feature").class_separation_F
    size = 10 + 30 * np.sqrt(F / F.max())
    n_comm = nodes.community.max()
    other_shown = False
    for cm in range(1, n_comm + 1):
        members = nodes[nodes.community == cm].feature.tolist()
        is_named = cm <= len(COMMUNITY_COLORS) and len(members) > 1
        name = f"Group {cm}" if is_named else "Other / unlinked"
        fig.add_scatter(
            x=[pos[n][0] for n in members], y=[pos[n][1] for n in members], mode="markers+text",
            marker=dict(size=[size[n] for n in members], color=COMMUNITY_COLORS[cm - 1] if is_named else OTHER,
                        line=dict(color="#fcfcfb", width=2)),
            text=members, textposition=["middle right" if pos[n][0] >= 0 else "middle left" for n in members],
            textfont=dict(size=10, color="#0b0b0b"), name=name, legendgroup=name,
            showlegend=is_named or not other_shown,
            customdata=np.c_[members, [F[n] for n in members], [G.degree(n) for n in members]],
            hovertemplate="<b>%{customdata[0]}</b><br>class separation F = %{customdata[1]:.0f}"
                          "<br>links: %{customdata[2]}<extra>" + name + "</extra>")
        other_shown = other_shown or not is_named
    fig.update_xaxes(visible=False, range=[-1.6, 1.6])
    fig.update_yaxes(visible=False, range=[-1.25, 1.25], scaleanchor="x")
    st.plotly_chart(style(fig, 740), width="stretch")
    top = F.sort_values(ascending=False).head(3).index.tolist()
    hub = nodes.sort_values("connections", ascending=False).feature.iloc[0]
    st.caption("Node size = how strongly the feature separates the five classes (ANOVA F-score). Node colour = group "
               "found by community detection (greedy modularity). Features are arranged in a circle by group; unlinked features come last. Lines are correlations above the "
               "threshold (dotted red = negative).")
    n_groups = int((nodes.groupby("community").size() > 1).sum())
    n_alone = int((nodes.connections == 0).sum())
    alt_text(f"Network of 38 features with {G.number_of_edges()} links at |r| ≥ {thr:.2f}, forming {n_groups} groups "
             f"(the {min(n_groups, len(COMMUNITY_COLORS))} largest in colour); {n_alone} features have no link. "
             f"Groups sit together as arcs of the circle. The most connected feature is {hub}. The features that best "
             f"separate classes (largest nodes) are {', '.join(top)}.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Links", G.number_of_edges())
    m2.metric("Groups (communities)", n_groups)
    m3.metric("Most connected feature", hub)
    maybe_table(nodes, "Show every feature with its group and scores")
    st.markdown(
        "**What we see:** the features fall into a few tightly linked groups. One tracks spectral shape (centroid, "
        "roll-off, zero-crossing rate), one is the MFCC variances that follow loudness swings (crest factor joins it, "
        "because clipping squashes those swings), and one is the middle spectral-contrast bands. Many of the 38 numbers "
        "repeat each other, so the classifier does not need all of them. Switch to *Narrow* and the groups get bigger and "
        "denser: with little variety, everything moves together."
    )
    st.warning(
        "**Shortcut alert for the ML project.** One of the largest nodes is **contrast7**, the highest frequency band. "
        "LibriSpeech was recorded at 16 kHz, so clean speech has *no* energy above 8 kHz, while the ESC-50 wind and "
        "noise recordings (44.1 kHz) and clipping distortion do. A model could learn \"anything above 8 kHz means a "
        "defect\" instead of learning the defect itself, and that rule would break on real phone recordings. "
        "Fix: low-pass or resample every clip to 16 kHz before extracting features. Look for the dark band at the top "
        "of the clean spectrograms in tab 3."
    )
    st.divider()

    # --- Class confusability network --------------------------------------------------------------
    st.markdown("**Question:** before training any model, which defects already *look alike* in feature space? "
                "Here each class is a node, and a thick link means the classes' clips sit next to each other.")
    c1, c2 = st.columns(2)
    s2 = c1.radio("Strategy", ["narrow", "diverse"], index=1, horizontal=True, key="nw_cls_strategy",
                  format_func=str.title)
    k = c2.slider("Neighbours per clip (k)", 5, 30, 10, key="nw_k")
    m = _class_graph(s2, k)
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(CLASSES), endpoint=False)
    xy = {c: (np.cos(a), np.sin(a)) for c, a in zip(CLASSES, ang)}
    fig = go.Figure()
    for i, a in enumerate(CLASSES):
        for b in CLASSES[i + 1:]:
            w = (m.loc[a, b] + m.loc[b, a]) / 2
            if w < 0.01:
                continue
            fig.add_scatter(x=[xy[a][0], xy[b][0]], y=[xy[a][1], xy[b][1]], mode="lines",
                            line=dict(color="#8a8984", width=1 + 40 * w), opacity=0.6, showlegend=False,
                            hoverinfo="skip")
            fig.add_annotation(x=(xy[a][0] + xy[b][0]) / 2, y=(xy[a][1] + xy[b][1]) / 2, text=f"{w:.0%}",
                               showarrow=False, font=dict(size=12, color="#0b0b0b"), bgcolor="#fcfcfb")
    for c in CLASSES:
        fig.add_scatter(x=[xy[c][0]], y=[xy[c][1]], mode="markers+text", text=[CLASS_LABELS[c]],
                        textposition="top center" if xy[c][1] >= 0 else "bottom center",
                        marker=dict(size=34, color=CLASS_COLORS[c], symbol=CLASS_SYMBOLS[c],
                                    line=dict(color="#fcfcfb", width=2)), name=CLASS_LABELS[c],
                        hovertemplate=f"<b>{CLASS_LABELS[c]}</b><br>{m.loc[c, c]:.0%} of neighbours are the same class"
                                      "<extra></extra>")
    fig.update_xaxes(visible=False, range=[-1.5, 1.5])
    fig.update_yaxes(visible=False, range=[-1.4, 1.4], scaleanchor="x")
    st.plotly_chart(style(fig, 500), width="stretch")
    pairs = [((m.loc[a, b] + m.loc[b, a]) / 2, a, b) for i, a in enumerate(CLASSES) for b in CLASSES[i + 1:]]
    w, a, b = max(pairs)
    alt_text(f"Five class nodes arranged in a circle, with link thickness and the printed percentage showing how often "
             f"a clip's nearest neighbours come from the other class. The thickest link is {CLASS_LABELS[a]} and "
             f"{CLASS_LABELS[b]} ({w:.0%}).")
    maybe_table((m * 100).round(1).rename(index=CLASS_LABELS, columns=CLASS_LABELS).reset_index(names="class"),
                "Show the neighbour-share matrix (%)")
    st.markdown(
        f"**What we see:** the strongest link is between {CLASS_LABELS[a]} and {CLASS_LABELS[b]}. Those are the classes "
        "the CSCI 6364 classifier is most likely to confuse, so its confusion matrix should show the same pair. "
        "Comparing strategies shows how adding variety moves the classes closer together. That makes the task harder in "
        "training, but closer to real audio."
    )
