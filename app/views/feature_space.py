"""Tab 4 (tabular / multivariate): distributions, a z-score heatmap, a pair plot and 2-D embeddings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from common import CLASS_COLORS, CLASS_LABELS, CLASSES, alt_text, feature_names, load_tables, maybe_table, style
from noisedx.config import CLASS_SYMBOLS, STRATEGY_COLORS
from noisedx.features import FEATURE_HELP

KEY_FEATURES = ["crest_factor", "flatness", "centroid", "zcr", "mfcc1_mean"]


@st.cache_data
def _table() -> pd.DataFrame:
    meta, feats = load_tables()
    return meta.merge(feats, on="id")


@st.cache_data
def _embed(method: str) -> pd.DataFrame:
    df = _table()
    X = StandardScaler().fit_transform(df[feature_names()])
    if method == "PCA":
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_
        cols = [f"PC1 ({ev[0]:.0%} of variance)", f"PC2 ({ev[1]:.0%} of variance)"]
    else:
        Z = TSNE(n_components=2, perplexity=35, random_state=0, init="pca").fit_transform(X)
        cols = ["t-SNE 1", "t-SNE 2"]
    out = df[["id", "label", "strategy", "split", "speaker"]].copy()
    out[cols[0]], out[cols[1]] = Z[:, 0], Z[:, 1]
    out.attrs["axes"] = cols
    return out


def render() -> None:
    st.markdown("**Question:** each clip becomes 38 numbers (the CSCI 6364 feature set). Do the classes separate in "
                "that table, and which numbers carry the signal?")
    df = _table()

    # --- Distribution of one feature ------------------------------------------------------------
    c1, c2 = st.columns([1, 2])
    feat = c1.selectbox("Feature", KEY_FEATURES + [f for f in feature_names() if f not in KEY_FEATURES],
                        key="fs_feat")
    c2.info(FEATURE_HELP.get(feat, "MFCCs summarise the overall shape of the spectrum. Low-numbered MFCCs capture "
                                   "broad tilt and loudness; higher ones capture finer detail. Spectral contrast "
                                   "compares peaks to valleys in each of 7 frequency bands."))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["Narrow", "Diverse"],
                        horizontal_spacing=0.03)
    for k, s in enumerate(["narrow", "diverse"], start=1):
        for c in CLASSES:
            d = df[(df.strategy == s) & (df.label == c)]
            fig.add_box(y=d[feat], name=CLASS_LABELS[c], marker_color=CLASS_COLORS[c], boxpoints="outliers",
                        line=dict(width=1.5), row=1, col=k, showlegend=False,
                        hovertemplate=f"{CLASS_LABELS[c]}<br>{feat} = %{{y:.3f}}<extra>{s}</extra>")
    fig.update_yaxes(title_text=feat, col=1)
    st.plotly_chart(style(fig, 380), width="stretch")
    med = df.groupby(["strategy", "label"], observed=True)[feat].median().unstack().round(3)
    lo_c, hi_c = med.loc["diverse"].idxmin(), med.loc["diverse"].idxmax()
    alt_text(f"Box plots of {feat} for each class, narrow strategy on the left and diverse on the right, "
             f"with the class named on the x-axis. In the diverse set the median is lowest for {CLASS_LABELS[lo_c]} "
             f"and highest for {CLASS_LABELS[hi_c]}. Diverse boxes are generally taller (more spread).")
    maybe_table(med.reset_index())
    st.divider()

    # --- Heatmap of class means (z-scores) ----------------------------------------------------------
    st.markdown("**Question:** across *all* 38 features at once, how does each class differ from the average clip?")
    s_hm = st.radio("Strategy", ["narrow", "diverse"], horizontal=True, key="fs_hm_strategy", format_func=str.title)
    sub = df[df.strategy == s_hm]
    z = (sub[feature_names()] - sub[feature_names()].mean()) / sub[feature_names()].std()
    zm = z.groupby(sub.label, observed=True).mean().T[CLASSES]
    fig = go.Figure(go.Heatmap(
        z=zm.values, x=[CLASS_LABELS[c] for c in CLASSES], y=zm.index, colorscale="RdBu_r", zmid=0, zmin=-2, zmax=2,
        colorbar=dict(title="z-score", thickness=12), xgap=2, ygap=1,
        hovertemplate="%{x}<br>%{y}: %{z:.2f} SD from the mean<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    st.plotly_chart(style(fig, 760), width="stretch")
    alt_text("Heatmap with 38 feature rows and 5 class columns. Red cells are above the average clip and blue cells "
             "are below, measured in standard deviations. Clipping is strongly blue on crest factor and the top contrast band, "
             "and red on the first MFCC (overall level). Noise is red on centroid, roll-off and zero-crossing rate. "
             "Wind and hum are light blue on centroid (darker, lower-pitched sound).")
    maybe_table(zm.round(2).reset_index(names="feature"))
    st.divider()

    # --- Pair plot ---------------------------------------------------------------------------------
    st.markdown("**Question:** do pairs of the physics-based features already split the classes?")
    s_pp = st.radio("Strategy", ["narrow", "diverse"], horizontal=True, key="fs_pp_strategy", format_func=str.title)
    pp = df[df.strategy == s_pp].assign(Class=lambda d: d.label.map(CLASS_LABELS))
    fig = px.scatter_matrix(pp, dimensions=["crest_factor", "flatness", "centroid", "zcr"], color="Class",
                            symbol="Class",
                            color_discrete_map={CLASS_LABELS[c]: CLASS_COLORS[c] for c in CLASSES},
                            symbol_map={CLASS_LABELS[c]: CLASS_SYMBOLS[c] for c in CLASSES},
                            category_orders={"Class": [CLASS_LABELS[c] for c in CLASSES]},
                            hover_data=["id"])
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=5, opacity=0.6))
    st.plotly_chart(style(fig, 640), width="stretch")
    alt_text("Pair plot of crest factor, flatness, centroid and zero-crossing rate, with each class in its own colour "
             "and marker shape. Clipping forms a separate group at low crest factor. Noise spreads toward high "
             "flatness and zero-crossing rate. Clean, hum and wind overlap heavily, so they need more features to tell apart.")
    st.divider()

    # --- 2-D embedding ------------------------------------------------------------------------------
    st.markdown("**Question:** if we squeeze all 38 features into 2 dimensions, do the classes form clusters, and does "
                "the diverse strategy cover more ground?")
    c1, c2 = st.columns(2)
    method = c1.radio("Method", ["PCA", "t-SNE"], horizontal=True, key="fs_method",
                      help="PCA is a linear projection (axes keep meaning). t-SNE preserves neighbourhoods, "
                           "but distances between clusters are not meaningful.")
    colour_by = c2.radio("Colour by", ["Class", "Strategy"], horizontal=True, key="fs_colour")
    emb = _embed(method)
    ax = emb.attrs.get("axes") or [c for c in emb.columns if c.startswith(("PC", "t-SNE"))]
    emb = emb.assign(Class=emb.label.map(CLASS_LABELS), Strategy=emb.strategy.str.title())
    if colour_by == "Class":
        fig = px.scatter(emb, x=ax[0], y=ax[1], color="Class", symbol="Class", facet_col="Strategy",
                         color_discrete_map={CLASS_LABELS[c]: CLASS_COLORS[c] for c in CLASSES},
                         symbol_map={CLASS_LABELS[c]: CLASS_SYMBOLS[c] for c in CLASSES},
                         category_orders={"Class": [CLASS_LABELS[c] for c in CLASSES], "Strategy": ["Narrow", "Diverse"]},
                         hover_data=["id", "speaker", "split"])
    else:
        fig = px.scatter(emb, x=ax[0], y=ax[1], color="Strategy", symbol="Strategy",
                         color_discrete_map={"Narrow": STRATEGY_COLORS["narrow"], "Diverse": STRATEGY_COLORS["diverse"]},
                         symbol_map={"Narrow": "circle", "Diverse": "diamond-open"},
                         category_orders={"Strategy": ["Narrow", "Diverse"]}, hover_data=["id", "Class", "split"])
    fig.update_traces(marker=dict(size=7, opacity=0.7, line=dict(width=0.5, color="#fcfcfb")))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(style(fig, 520), width="stretch")
    alt_text(f"{method} scatter plot of all 1,500 clips, narrow on the left and diverse on the right. In the narrow panel, "
             "noise and clipping form tight groups of their own and clean, hum and wind sit near each other. In the diverse "
             "panel every class spreads out and the groups overlap much more. Coloured by strategy, the diverse points "
             "cover a wider area around the narrow clusters.")

    # Spread metric: mean distance to class centroid in standardised feature space
    X = StandardScaler().fit_transform(df[feature_names()])
    spread = []
    for s in ["narrow", "diverse"]:
        for c in CLASSES:
            m = ((df.strategy == s) & (df.label == c)).to_numpy()
            spread.append({"strategy": s, "class": c,
                           "spread": float(np.linalg.norm(X[m] - X[m].mean(axis=0), axis=1).mean())})
    sp = pd.DataFrame(spread)
    fig = go.Figure()
    for s in ["narrow", "diverse"]:
        d = sp[sp.strategy == s]
        fig.add_bar(x=d["class"].map(CLASS_LABELS), y=d.spread, name=s.title(), marker_color=STRATEGY_COLORS[s],
                    marker_line=dict(color="#fcfcfb", width=2), text=d.spread.round(1), textposition="outside",
                    hovertemplate="%{x}<br>" + s.title() + ": %{y:.2f}<extra></extra>")
    fig.update_layout(barmode="group", yaxis_title="Average distance to class centre (SD units)",
                      yaxis_range=[0, sp.spread.max() * 1.2])
    st.plotly_chart(style(fig, 360, "How spread out is each class? (all 38 features, standardised)"), width="stretch")
    ratio = sp.pivot(index="class", columns="strategy", values="spread")
    alt_text("Grouped bar chart of within-class spread. For every class, the diverse bar is taller than the narrow "
             f"bar (on average {(ratio.diverse / ratio.narrow).mean():.1f} times as spread out).")
    maybe_table(sp.round(2))
    st.markdown(
        "**What we see:** with narrow data, noise and clipping separate cleanly and the task looks easy. With diverse "
        "data the classes overlap far more, because each class now covers many strengths, rooms and sources. "
        "Within-class spread grows for every class. That is the domain-randomisation bet in H2: a harder, wider "
        "training cloud is more likely to contain the real recordings the model will meet later."
    )
