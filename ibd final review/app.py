"""
========================================================================================
Evaluating Generalization Collapse in ML-based PPI Prediction:
A Comparative Study Across Random vs. Protein-Disjoint Data Splits
========================================================================================
Interactive Academic Research Dashboard
First-Year College Research Project | Computational Biology & Machine Learning
Framework: Python Streamlit & Plotly
========================================================================================
"""

import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & METADATA
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="PPI Generalization Collapse | Academic Research Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------------
# 2. FILE RESOLUTION & DATA CACHING (Graceful handling of paths & missing files)
# --------------------------------------------------------------------------------------
def resolve_file_path(relative_path: str):
    """
    Finds file path across possible execution directories.
    Handles running from root, ppi_project, or workspace seamlessly.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppi_project", relative_path),
        os.path.join(os.getcwd(), relative_path),
        os.path.join(os.getcwd(), "ppi_project", relative_path),
        os.path.join(r"c:\Users\kathi\Downloads\ibd final review\ppi_project", relative_path),
        os.path.join(r"c:\Users\kathi\Downloads\ibd final review", relative_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None

@st.cache_data(show_spinner=False)
def load_csv_data(relative_path: str, sep: str = ",", header="infer", index_col=None, dtype=None):
    """Safely loads a CSV file and caches it."""
    resolved = resolve_file_path(relative_path)
    if resolved and os.path.exists(resolved):
        try:
            return pd.read_csv(resolved, sep=sep, header=header, index_col=index_col, dtype=dtype)
        except Exception as e:
            st.error(f"Error loading {relative_path}: {e}")
            return None
    return None

@st.cache_data(show_spinner=False)
def load_image_asset(relative_path: str):
    """Safely loads an image asset and caches it."""
    resolved = resolve_file_path(relative_path)
    if resolved and os.path.exists(resolved):
        try:
            return Image.open(resolved)
        except Exception as e:
            st.error(f"Error loading image {relative_path}: {e}")
            return None
    return None

# --------------------------------------------------------------------------------------
# 3. CUSTOM STYLING (Dark Navy & Blue Palette, Modern Typography, Cards & Badges)
# --------------------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Base theme colors & typography */
:root {
    --bg-navy-dark: #0A1128;
    --bg-navy-card: #131E3A;
    --bg-slate-card: #1E293B;
    --border-card: #2E3E5B;
    --blue-accent: #38BDF8;
    --blue-primary: #2563EB;
    --blue-glow: rgba(56, 189, 248, 0.15);
    --coral-accent: #F43F5E;
    --emerald-accent: #10B981;
    --amber-accent: #F59E0B;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
}

/* Background overrides */
.stApp {
    background-color: #0B132B;
    color: #F8FAFC;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #080D1A !important;
    border-right: 1px solid #1E293B;
}

/* Header Banner */
.hero-container {
    background: linear-gradient(135deg, #131E3A 0%, #0F172A 50%, #0B132B 100%);
    border: 1px solid #2A3B5C;
    border-radius: 12px;
    padding: 26px 30px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(56, 189, 248, 0.1);
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.3;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    line-height: 1.5;
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-right: 6px;
    margin-bottom: 6px;
}

.badge-blue {
    background-color: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.35);
}

.badge-coral {
    background-color: rgba(244, 63, 94, 0.15);
    color: #FB7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
}

.badge-emerald {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-amber {
    background-color: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

/* Metric Cards */
.metric-box {
    background-color: #131E3A;
    border: 1px solid #2A3B5C;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-box:hover {
    transform: translateY(-2px);
    border-color: #38BDF8;
}

.metric-title {
    font-size: 0.85rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.2;
}

.metric-subtitle {
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 4px;
}

.metric-delta-neg {
    color: #FB7185;
    font-weight: 600;
    font-size: 0.85rem;
}

.metric-delta-pos {
    color: #34D399;
    font-weight: 600;
    font-size: 0.85rem;
}

/* Research Content Card */
.research-card {
    background-color: #111A33;
    border: 1px solid #253350;
    border-radius: 10px;
    padding: 22px 24px;
    margin-bottom: 20px;
}

.research-card-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #38BDF8;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Callout Box for Scientific Rigor */
.scientific-callout {
    background-color: rgba(30, 41, 59, 0.7);
    border-left: 4px solid #F59E0B;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 18px 0;
}

.scientific-callout-blue {
    background-color: rgba(19, 30, 58, 0.7);
    border-left: 4px solid #38BDF8;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 18px 0;
}

.scientific-callout-rose {
    background-color: rgba(30, 20, 35, 0.7);
    border-left: 4px solid #F43F5E;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 18px 0;
}

/* Clean code / monospace highlights */
.mono-pill {
    font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    background-color: rgba(148, 163, 184, 0.12);
    color: #38BDF8;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.88em;
}

/* Plotly chart container */
.plotly-container {
    background-color: #131E3A;
    border: 1px solid #2A3B5C;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 18px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & PROJECT METADATA
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 PPI Research Lab")
    st.caption("Evaluating Generalization Collapse in ML-based PPI Prediction")
    st.markdown("---")

    selected_page = st.radio(
        "Navigate Dashboard",
        [
            "🏠 1. Executive Summary & Overview",
            "📊 2. SHS27K Dataset & Binary Formulation",
            "🔬 3. Methodology & 40D Feature Space",
            "📈 4. Results & Generalization Collapse",
            "🔍 5. 20-Seed Experiment Explorer",
            "🎯 6. Conclusions, Limitations & Future Work"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 📌 Benchmark Metadata")
    st.markdown("""
    - **Dataset**: `SHS27K (STRING v10)`
    - **Taxonomy**: *Homo sapiens (9606)*
    - **Features**: `40D AAC Symmetrical`
    - **Model**: `L2 Logistic Regression`
    - **Replicates**: `20 Random vs 20 Disjoint`
    - **Project Type**: First-Year College Research
    """)

    st.markdown("---")
    st.markdown("#### 📁 File Verification Status")
    key_files = [
        ("raw actions", "data/raw/SHS27k.actions.txt"),
        ("raw seqs", "data/raw/SHS27k.seqs.tsv"),
        ("positive pairs", "processed/positive_pairs.csv"),
        ("negative pairs", "processed/negative_pairs_random.csv"),
        ("binary dataset", "processed/binary_dataset_random.csv"),
        ("20-seed results", "results/twenty_seed_results.csv"),
        ("20-seed summary", "results/twenty_seed_summary.csv"),
        ("accuracy graph", "results/accuracy_comparison.png"),
        ("roc-auc graph", "results/roc_auc_comparison.png"),
    ]
    all_found = True
    for label, path in key_files:
        p = resolve_file_path(path)
        if p:
            st.markdown(f"<span style='color:#10B981;font-size:0.8rem;'>●</span> <span style='font-size:0.8rem;color:#CBD5E1;'>{label}</span>", unsafe_allow_html=True)
        else:
            all_found = False
            st.markdown(f"<span style='color:#EF4444;font-size:0.8rem;'>●</span> <span style='font-size:0.8rem;color:#F87171;'>{label} (missing)</span>", unsafe_allow_html=True)
    if all_found:
        st.markdown("<div style='margin-top:8px;'><span class='badge badge-emerald'>All 100% Verified</span></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# SHARED THEME & COLOR UTILITIES FOR PLOTLY
# --------------------------------------------------------------------------------------
PLOT_BG = "#101932"
PAPER_BG = "#101932"
GRID_COLOR = "#22314E"
COLOR_RANDOM = "#38BDF8"      # Bright sky cyan
COLOR_DISJOINT = "#F43F5E"    # Vibrant coral/rose

def apply_plotly_theme(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#F8FAFC", size=15, family="sans-serif")),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#94A3B8", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        height=height,
        legend=dict(
            bgcolor="rgba(19, 30, 58, 0.7)",
            bordercolor="#2A3B5C",
            borderwidth=1,
            font=dict(color="#E2E8F0")
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig

# ======================================================================================
# PAGE 1: EXECUTIVE SUMMARY & OVERVIEW
# ======================================================================================
if "1. Executive Summary" in selected_page:
    st.markdown("""
    <div class="hero-container">
        <div>
            <span class="badge badge-blue">Academic Research Project</span>
            <span class="badge badge-coral">Computational Biology</span>
            <span class="badge badge-emerald">Machine Learning Evaluation</span>
            <span class="badge badge-amber">20-Seed Replicated Benchmark</span>
        </div>
        <div class="hero-title">
            Evaluating Generalization Collapse in ML-based PPI Prediction:
            A Comparative Study Across Random vs. Protein-Disjoint Data Splits
        </div>
        <div class="hero-subtitle">
            An empirical investigation into data leakage, node memorization, and performance degradation in sequence-based protein-protein interaction classifiers using the benchmark SHS27K dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Random Split Accuracy</div>
            <div class="metric-value" style="color:#38BDF8;">61.39%</div>
            <div class="metric-subtitle">Mean ± 0.92% std (20 Seeds)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Disjoint Split Accuracy</div>
            <div class="metric-value" style="color:#FB7185;">58.13%</div>
            <div class="metric-subtitle"><span class="metric-delta-neg">▼ -3.26% Collapse</span> (± 3.45% std)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Random ROC-AUC</div>
            <div class="metric-value" style="color:#38BDF8;">0.6619</div>
            <div class="metric-subtitle">Mean ± 0.0106 std (20 Seeds)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Disjoint ROC-AUC</div>
            <div class="metric-value" style="color:#FB7185;">0.6151</div>
            <div class="metric-subtitle"><span class="metric-delta-neg">▼ -0.0468 Collapse</span> (± 0.0426 std)</div>
        </div>
        """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div class="research-card">
            <div class="research-card-header">
                🔬 The Research Problem: The Generalization Dilemma in PPI Prediction
            </div>
            <p style="color:#CBD5E1; line-height:1.6;">
                Protein-Protein Interactions (PPIs) orchestrate virtually all cellular processes, including signal transduction, metabolic regulation, and immune cascades. Elucidating the interactome is essential for dissecting disease etiologies and accelerating target discovery. However, experimental methods such as Yeast Two-Hybrid (Y2H) and Affinity Purification-Mass Spectrometry (AP-MS) are labor-intensive, costly, and subject to high false-negative rates.
            </p>
            <p style="color:#CBD5E1; line-height:1.6;">
                While machine learning on sequence features has emerged as a promising high-throughput alternative, <b>standard academic evaluation practices suffer from a pervasive structural flaw</b>: evaluating models on random edge-level splits rather than strict protein-disjoint splits.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="research-card">
            <div class="research-card-header">
                ⚠️ Why Random Train-Test Splitting Overestimates Performance
            </div>
            <p style="color:#CBD5E1; line-height:1.6;">
                When an interaction graph is split randomly by pairs (edges), individual proteins (nodes) are distributed across both the training set and the test set. According to the foundational evaluation taxonomy formulated by Park & Marcotte (2012):
            </p>
            <ul style="color:#CBD5E1; line-height:1.7;">
                <li><b>C1 (Both Proteins Seen)</b>: Both interacting partners in the test pair were present during model training.</li>
                <li><b>C2 (One Protein Seen)</b>: One partner was observed during training; the other is novel.</li>
                <li><b>C3 (Both Proteins Unseen)</b>: Neither partner was ever observed during training (pure out-of-sample generalization).</li>
            </ul>
            <p style="color:#CBD5E1; line-height:1.6;">
                <b>The Fatal Flaw:</b> In random 80:20 edge splits, over 90% of test pairs fall into categories <b>C1 and C2</b>. Consequently, the model does not learn universal biophysical interaction principles; instead, it memorizes individual protein representations, high-degree hubs, and dataset biases. When deployed on genuinely novel proteins (C3), prediction accuracy collapses.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="research-card">
            <div class="research-card-header">
                🎯 Core Project Objectives
            </div>
            <ol style="color:#CBD5E1; line-height:1.7; padding-left:20px;">
                <li><b>Standardized Benchmark Setup:</b> Curate a balanced binary dataset from the benchmark human SHS27K (STRING v10) repository (15,248 total pairs).</li>
                <li><b>Physicochemical Feature Space:</b> Extract normalized Amino Acid Composition (AAC) and formulate an order-invariant 40-dimensional pair representation.</li>
                <li><b>Strict Disjoint Formulation:</b> Implement a node-disjoint partitioning algorithm with strict crossing-edge pruning to prevent information leakage.</li>
                <li><b>20-Seed Statistical Replication:</b> Quantify empirical performance drop and evaluate the surge in variance across 20 distinct random seeds.</li>
                <li><b>Transparent Academic Assessment:</b> Objectively document experimental limitations without inflated marketing claims.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="scientific-callout">
            <b style="color:#F59E0B;">🎓 Academic Integrity & Scientific Realism:</b><br>
            <span style="color:#CBD5E1; font-size:0.9rem; line-height:1.5;">
            In contrast to flawed claims in literature alleging 90–95% accuracy on simplistic sequence features, our rigorous evaluation demonstrates that a linear model on global composition achieves realistic ~58–61% accuracy. Furthermore, negative pairs are sampled unobserved pairs, not experimentally confirmed non-interactions.
            </span>
        </div>
        """, unsafe_allow_html=True)

# ======================================================================================
# PAGE 2: DATASET OVERVIEW & BINARY FORMULATION
# ======================================================================================
elif "2. SHS27K Dataset" in selected_page:
    st.markdown("""
    <div class="hero-container">
        <span class="badge badge-blue">SHS27K Benchmark</span>
        <span class="badge badge-emerald">STRING v10 Database</span>
        <div class="hero-title">SHS27K Benchmark Dataset & Binary Formulation</div>
        <div class="hero-subtitle">
            Inspection of human interaction networks, multi-modal interaction actions, sequence completeness, and balanced binary dataset construction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Raw Records</div>
            <div class="metric-value">63,408</div>
            <div class="metric-subtitle">STRING Actions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Unique Proteins</div>
            <div class="metric-value">1,690</div>
            <div class="metric-subtitle">Homo sapiens (9606)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Sequences</div>
            <div class="metric-value">1,690</div>
            <div class="metric-subtitle">100% FASTA Match</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Positive Pairs</div>
            <div class="metric-value">7,624</div>
            <div class="metric-subtitle">Unique Undirected</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Sampled Negatives</div>
            <div class="metric-value">7,624</div>
            <div class="metric-subtitle">Random Non-Interacting</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Binary Dataset</div>
            <div class="metric-value">15,248</div>
            <div class="metric-subtitle">1:1 Balanced Ratio</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="scientific-callout">
        <b style="color:#F59E0B; font-size:1.05rem;">⚠️ Critical Distinction: Sampled Negatives vs. Confirmed Non-Interactors</b>
        <p style="color:#CBD5E1; margin-top:6px; line-height:1.55; font-size:0.92rem;">
            In computational biology, <b>experimentally confirmed non-interacting protein pairs virtually do not exist</b> due to publication bias (negative results are rarely published) and assay constraints (inability to test all conditions/compartments).
            In this study, negative pairs are generated via uniform random non-interacting pairing from the 1,690 proteins, strictly excluding known positive interactions and self-interactions.
            <b>These pairs are statistically sampled pseudo-negatives, not biologically verified non-interacting proteins.</b> A fraction may represent true undiscovered interactions (false negatives).
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_exp = st.columns([3, 2])

    with col_chart:
        st.markdown("#### 🔬 Distribution of STRING Interaction Modes (63,408 Raw Records)")
        modes_data = pd.DataFrame({
            "Mode": ["Reaction", "Binding", "Catalysis", "Activation", "Inhibition", "Ptmod", "Expression"],
            "Count": [18162, 16056, 11796, 7400, 5550, 2872, 1572]
        })
        fig_modes = px.bar(
            modes_data,
            x="Count",
            y="Mode",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale=[[0, "#1E3A8A"], [1, "#38BDF8"]],
            title="Raw Action Modes in SHS27k.actions.txt"
        )
        fig_modes.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_modes = apply_plotly_theme(fig_modes, title="Raw Action Modes in SHS27k.actions.txt", height=340)
        fig_modes.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_modes, use_container_width=True)

    with col_exp:
        st.markdown("#### ⚙️ Data Preparation Pipeline")
        st.markdown("""
        <div class="research-card">
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                <b>Step 1: Multi-Record De-duplication</b><br>
                Many protein pairs have multiple recorded actions (e.g., binding + catalysis). The raw 63,408 records are aggregated into order-independent pairs:
                <br><span class="mono-pill">protein_a = min(id1, id2), protein_b = max(id1, id2)</span><br>
                Yielding exactly <b>7,624 unique interacting pairs</b>.
            </p>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                <b>Step 2: Sequence Completeness Verification</b><br>
                All 1,690 interacting proteins match 100% of the entries in <span class="mono-pill">SHS27k.seqs.tsv</span>. Zero missing sequences detected.
            </p>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                <b>Step 3: Balanced Binary Formulation</b><br>
                7,624 pseudo-negative pairs are generated with random seed 42 to produce a perfectly balanced 15,248 pair binary dataset (50% positive / 50% negative).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📂 Interactive Processed Data Inspector")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Positive Pairs (7,624)",
        "Sampled Negatives (7,624)",
        "Binary Dataset (15,248)",
        "Pair Modes Summary"
    ])

    with tab1:
        df_pos = load_csv_data("processed/positive_pairs.csv")
        if df_pos is not None:
            st.dataframe(df_pos.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(df_pos):,} verified positive interaction pairs.")
        else:
            st.warning("processed/positive_pairs.csv could not be located.")

    with tab2:
        df_neg = load_csv_data("processed/negative_pairs_random.csv")
        if df_neg is not None:
            st.dataframe(df_neg.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(df_neg):,} sampled negative pairs (pseudo-negatives).")
        else:
            st.warning("processed/negative_pairs_random.csv could not be located.")

    with tab3:
        df_bin = load_csv_data("processed/binary_dataset_random.csv")
        if df_bin is not None:
            st.dataframe(df_bin.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(df_bin):,} balanced pairs in binary_dataset_random.csv.")
        else:
            st.warning("processed/binary_dataset_random.csv could not be located.")

    with tab4:
        df_summary = load_csv_data("results/pair_summary.csv")
        if df_summary is not None:
            st.dataframe(df_summary.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(df_summary):,} unique pairs with interaction mode annotations.")
        else:
            st.warning("results/pair_summary.csv could not be located.")

# ======================================================================================
# PAGE 3: METHODOLOGY & 40D FEATURE SPACE
# ======================================================================================
elif "3. Methodology" in selected_page:
    st.markdown("""
    <div class="hero-container">
        <span class="badge badge-blue">Feature Engineering</span>
        <span class="badge badge-emerald">Logistic Regression</span>
        <span class="badge badge-coral">Evaluation Topologies</span>
        <div class="hero-title">Methodology: 40D Feature Space & Splitting Protocols</div>
        <div class="hero-subtitle">
            Mathematical derivation of Amino Acid Composition (AAC), symmetric order-invariant pair features, and the topological contrast between Random vs. Protein-Disjoint evaluation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1. Amino Acid Composition (AAC) Representation")
    col_aac_text, col_aac_calc = st.columns([3, 2])

    with col_aac_text:
        st.markdown(r"""
        <div class="research-card">
            <p style="color:#CBD5E1; line-height:1.6;">
                Each protein sequence is mapped to a <b>20-dimensional normalized frequency vector</b> capturing the relative presence of the 20 canonical amino acids:
            </p>
            <div style="background-color:#080D1A; padding:12px 18px; border-radius:8px; border:1px solid #1E293B; margin:12px 0;">
                $$	ext{AAC}_i = rac{	ext{Count of amino acid } i}{	ext{Total valid standard amino acids in sequence}}, \quad i \in \{A, C, D, \dots, Y\}$$
            </div>
            <p style="color:#CBD5E1; line-height:1.6;">
                AAC condenses sequence information into bulk physicochemical attributes: hydrophobicity (e.g., Leu, Ile, Val), charge (e.g., Lys, Arg vs. Asp, Glu), polarity, and aromaticity (e.g., Trp, Phe, Tyr).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_aac_calc:
        st.markdown("#### 🧪 Interactive AAC Explorer")
        sample_choice = st.selectbox(
            "Select an authentic SHS27K protein:",
            [
                "9606.ENSP00000000233 (ARF5 - 180 aa)",
                "9606.ENSP00000250971 (Insulin - 110 aa)",
                "9606.ENSP00000019317 (PTPRF - 655 aa)"
            ]
        )
        sample_seqs = {
            "9606.ENSP00000000233 (ARF5 - 180 aa)": "MGLTVSALFSRIFGKKQMRILMVGLDAAGKTTILYKLKLGEIVTTIPTIGFNVETVEYKNISFTVWDVGGQDKIRPLWRHYFQNTQGLIFVVDSNDRERVNEAREELMRMLAEDELRDAVLLVFANKQDLPNAMNAAEITDKLGLHSLRHRNWYIQATCATSGDGLYEGLDWLSNQLRNQK",
            "9606.ENSP00000250971 (Insulin - 110 aa)": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
            "9606.ENSP00000019317 (PTPRF - 655 aa)": "MTECFLPPTSSPSEHRRVEHGSGLTRTPSSEEISPTKFPGLYRTSMAPATSRPVPLSRFLLLSVLLLLSSAAACSVDTSPPVSALAGRPRLYSFQSNPNPAVRLRLLSQ"
        }
        active_seq = sample_seqs[sample_choice]
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        counts = [active_seq.count(aa) for aa in aa_alphabet]
        freqs = [c / len(active_seq) for c in counts]

        fig_aac = px.bar(
            x=list(aa_alphabet),
            y=freqs,
            labels={"x": "Amino Acid", "y": "Frequency"},
            title=f"AAC Profile ({len(active_seq)} Residues)"
        )
        fig_aac.update_traces(marker_color="#38BDF8")
        fig_aac = apply_plotly_theme(fig_aac, title=f"AAC Profile ({len(active_seq)} Residues)", height=220)
        st.plotly_chart(fig_aac, use_container_width=True)

    st.markdown("### 2. 40-Dimensional Symmetrical Pair Feature Vector")
    st.markdown(r"""
    <div class="research-card">
        <p style="color:#CBD5E1; line-height:1.6;">
            Protein-protein binding in a binary classification setting is <b>biophysically undirected</b>: the interaction between Protein A and Protein B is identical to the interaction between Protein B and Protein A:
            <span class="mono-pill">Pair(A, B) ≡ Pair(B, A)</span>.
            To enforce mathematical invariance to order, each candidate pair is mapped into a <b>40-dimensional composite vector</b>:
        </p>
        <div style="background-color:#080D1A; padding:16px 20px; border-radius:8px; border:1px solid #1E293B; margin:14px 0;">
            $$\mathbf{x}_{(A, B)} = \Big[ |\mathbf{v}_A - \mathbf{v}_B| \;,\; \mathbf{v}_A \odot \mathbf{v}_B \Big] \in \mathbb{R}^{40}$$
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:10px;">
            <div style="background-color:#131E3A; padding:12px 16px; border-radius:6px; border:1px solid #2A3B5C;">
                <b style="color:#38BDF8;">1. Absolute Difference (20D):</b><br>
                <span style="color:#CBD5E1; font-size:0.9rem;">Measures compositional distance and biophysical disparity (e.g. difference in hydrophobicity, electrostatic polarity mismatch). Symmetric: $|v_A - v_B| = |v_B - v_A|$.</span>
            </div>
            <div style="background-color:#131E3A; padding:12px 16px; border-radius:6px; border:1px solid #2A3B5C;">
                <b style="color:#38BDF8;">2. Hadamard Product (20D):</b><br>
                <span style="color:#CBD5E1; font-size:0.9rem;">Captures compositional co-occurrence (e.g. both proteins enriched in Cysteines for disulfide bonding or Leucines for coiled-coil zippers). Symmetric: $v_A \odot v_B = v_B \odot v_A$.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 3. Model Architecture & Preprocessing Integrity")
    col_mod1, col_mod2 = st.columns(2)
    with col_mod1:
        st.markdown(r"""
        <div class="research-card">
            <div class="research-card-header">🛡️ Standard Scaling Without Leakage</div>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                A <span class="mono-pill">StandardScaler</span> is fitted <b>exclusively on the training feature matrix</b> ($X_{train}$) and subsequently applied to transform test pairs ($X_{test}$):
                <br><br>
                $$\mu_{train}, \sigma_{train} \leftarrow 	ext{fit}(X_{train}); \quad X_{test}^{scaled} = rac{X_{test} - \mu_{train}}{\sigma_{train}}$$
                Fitting the scaler on the entire dataset would cause mean/variance leakage between partitions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_mod2:
        st.markdown(r"""
        <div class="research-card">
            <div class="research-card-header">⚖️ Regularized Logistic Regression</div>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                We employ an $L_2$-regularized Logistic Regression model (<span class="mono-pill">max_iter=1000, random_state=seed</span>):
                <br><br>
                $$P(y=1|\mathbf{x}) = rac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
                <b>Why Logistic Regression?</b> It provides a convex, highly interpretable baseline. It eliminates confounding hyperparameter over-fitting, cleanly isolating the statistical effect of data splitting.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 4. Splitting Topology: Random vs. Protein-Disjoint")
    st.markdown("""
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-top:12px;">
        <div style="background-color:#131E3A; border:1px solid #38BDF8; border-radius:10px; padding:20px;">
            <div style="color:#38BDF8; font-size:1.15rem; font-weight:700; margin-bottom:8px;">
                Strategy A: Random 80:20 Split (Edge Partition)
            </div>
            <svg viewBox="0 0 400 180" style="width:100%; height:auto;">
                <rect width="400" height="180" fill="#0A1128" rx="8" />
                <circle cx="80" cy="50" r="14" fill="#38BDF8" />
                <circle cx="200" cy="40" r="14" fill="#38BDF8" />
                <circle cx="320" cy="50" r="14" fill="#38BDF8" />
                <circle cx="120" cy="130" r="14" fill="#38BDF8" />
                <circle cx="260" cy="130" r="14" fill="#38BDF8" />
                <line x1="80" y1="50" x2="200" y2="40" stroke="#38BDF8" stroke-width="3" />
                <line x1="200" y1="40" x2="320" y2="50" stroke="#38BDF8" stroke-width="3" />
                <line x1="80" y1="50" x2="120" y2="130" stroke="#F43F5E" stroke-width="3" stroke-dasharray="4" />
                <line x1="200" y1="40" x2="260" y2="130" stroke="#38BDF8" stroke-width="3" />
                <line x1="120" y1="130" x2="260" y2="130" stroke="#F43F5E" stroke-width="3" stroke-dasharray="4" />
                <text x="20" y="168" fill="#94A3B8" font-size="11">Solid blue: Train Edges (80%)</text>
                <text x="210" y="168" fill="#FB7185" font-size="11">Dashed pink: Test Edges (20%)</text>
            </svg>
            <div style="color:#CBD5E1; font-size:0.88rem; margin-top:12px; line-height:1.5;">
                ● <b>Mechanism</b>: Edges are randomly selected into train/test.<br>
                ● <b>Node Leakage</b>: High. Protein nodes appear in both sets.<br>
                ● <b>Classes</b>: >90% C1 (both seen) and C2 (one seen).<br>
                ● <b>Consequence</b>: Model memorizes frequent protein nodes.
            </div>
        </div>

        <div style="background-color:#131E3A; border:1px solid #F43F5E; border-radius:10px; padding:20px;">
            <div style="color:#FB7185; font-size:1.15rem; font-weight:700; margin-bottom:8px;">
                Strategy B: Protein-Disjoint 80:20 Split (Node Partition)
            </div>
            <svg viewBox="0 0 400 180" style="width:100%; height:auto;">
                <rect width="400" height="180" fill="#0A1128" rx="8" />
                <rect x="15" y="20" width="180" height="120" fill="rgba(56,189,248,0.08)" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="4" rx="8" />
                <text x="25" y="38" fill="#38BDF8" font-size="11" font-weight="bold">Train Proteins V_train (80%)</text>
                <circle cx="50" cy="75" r="12" fill="#38BDF8" />
                <circle cx="150" cy="70" r="12" fill="#38BDF8" />
                <circle cx="95" cy="115" r="12" fill="#38BDF8" />
                <line x1="50" y1="75" x2="150" y2="70" stroke="#38BDF8" stroke-width="2.5" />
                <line x1="50" y1="75" x2="95" y2="115" stroke="#38BDF8" stroke-width="2.5" />

                <rect x="235" y="20" width="150" height="120" fill="rgba(244,63,94,0.08)" stroke="#F43F5E" stroke-width="1.5" stroke-dasharray="4" rx="8" />
                <text x="245" y="38" fill="#FB7185" font-size="11" font-weight="bold">Test V_test (20%)</text>
                <circle cx="275" cy="85" r="12" fill="#F43F5E" />
                <circle cx="345" cy="85" r="12" fill="#F43F5E" />
                <line x1="275" y1="85" x2="345" y2="85" stroke="#F43F5E" stroke-width="2.5" />

                <line x1="150" y1="70" x2="275" y2="85" stroke="#64748B" stroke-width="2" stroke-dasharray="2" />
                <text x="180" y="105" fill="#EF4444" font-size="14" font-weight="bold">✕</text>
                <text x="20" y="168" fill="#94A3B8" font-size="11">Crossing pairs pruned to eliminate data leakage.</text>
            </svg>
            <div style="color:#CBD5E1; font-size:0.88rem; margin-top:12px; line-height:1.5;">
                ● <b>Mechanism</b>: Proteins are partitioned with zero intersection.<br>
                ● <b>Node Leakage</b>: Zero. Test proteins are strictly unseen.<br>
                ● <b>Classes</b>: 100% C3 (both proteins novel).<br>
                ● <b>Consequence</b>: Unbiased test of true biochemical generalization.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================================================
# PAGE 4: RESULTS & COMPARATIVE ANALYSIS
# ======================================================================================
elif "4. Results" in selected_page:
    st.markdown("""
    <div class="hero-container">
        <span class="badge badge-coral">Generalization Collapse</span>
        <span class="badge badge-blue">Statistical Analysis</span>
        <span class="badge badge-amber">Standard Deviation Surge</span>
        <div class="hero-title">Empirical Results: Random vs. Protein-Disjoint Evaluation</div>
        <div class="hero-subtitle">
            Quantitative proof of generalization collapse: comparative analysis across 20 independent seeds, standard deviation expansion, and original research graphs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Mean Accuracy</div>
            <div class="metric-value">
                <span style="color:#38BDF8; font-size:1.3rem;">61.39%</span> <span style="font-size:0.9rem; color:#94A3B8;">vs</span> <span style="color:#FB7185; font-size:1.3rem;">58.13%</span>
            </div>
            <div class="metric-subtitle"><span class="metric-delta-neg">▼ -3.26% Drop</span> (p < 0.001)</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Mean ROC-AUC</div>
            <div class="metric-value">
                <span style="color:#38BDF8; font-size:1.3rem;">0.6619</span> <span style="font-size:0.9rem; color:#94A3B8;">vs</span> <span style="color:#FB7185; font-size:1.3rem;">0.6151</span>
            </div>
            <div class="metric-subtitle"><span class="metric-delta-neg">▼ -0.0468 Drop</span> (p < 0.001)</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Accuracy Std. Dev.</div>
            <div class="metric-value">
                <span style="color:#38BDF8; font-size:1.3rem;">0.92%</span> <span style="font-size:0.9rem; color:#94A3B8;">→</span> <span style="color:#FB7185; font-size:1.3rem;">3.45%</span>
            </div>
            <div class="metric-subtitle"><span class="metric-delta-neg">▲ 3.76× Variance Surge</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">ROC-AUC Std. Dev.</div>
            <div class="metric-value">
                <span style="color:#38BDF8; font-size:1.3rem;">0.0106</span> <span style="font-size:0.9rem; color:#94A3B8;">→</span> <span style="color:#FB7185; font-size:1.3rem;">0.0426</span>
            </div>
            <div class="metric-subtitle"><span class="metric-delta-neg">▲ 4.02× Variance Surge</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 1. Generated Research Graphs (Matplotlib)")
    col_img1, col_img2 = st.columns(2)

    img_acc = load_image_asset("results/accuracy_comparison.png")
    img_auc = load_image_asset("results/roc_auc_comparison.png")

    with col_img1:
        if img_acc:
            st.image(img_acc, caption="Figure 1: Mean Accuracy Comparison with Error Bars (±1 Std. Dev.)", use_column_width=True)
        else:
            st.warning("results/accuracy_comparison.png not found.")

    with col_img2:
        if img_auc:
            st.image(img_auc, caption="Figure 2: Mean ROC-AUC Comparison with Error Bars (±1 Std. Dev.)", use_column_width=True)
        else:
            st.warning("results/roc_auc_comparison.png not found.")

    st.markdown("### 2. Comprehensive 20-Seed Statistical Summary")
    stats_table = pd.DataFrame({
        "Evaluation Metric": ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1-Score"],
        "Random Mean ± Std": [
            "61.39% ± 0.92%",
            "0.6619 ± 0.0106",
            "61.00% ± 0.87%",
            "63.17% ± 1.38%",
            "62.06% ± 1.01%"
        ],
        "Disjoint Mean ± Std": [
            "58.13% ± 3.45%",
            "0.6151 ± 0.0426",
            "58.05% ± 3.11%",
            "57.72% ± 7.68%",
            "57.75% ± 5.16%"
        ],
        "Absolute Change (Δ)": [
            "-3.26 percentage pts",
            "-0.0468",
            "-2.95 percentage pts",
            "-5.45 percentage pts",
            "-4.31 percentage pts"
        ],
        "Std. Dev. Ratio (Disjoint / Random)": [
            "3.76× wider spread",
            "4.02× wider spread",
            "3.56× wider spread",
            "5.58× wider spread",
            "5.10× wider spread"
        ]
    })
    st.dataframe(stats_table, use_container_width=True, hide_index=True)

    st.markdown("### 3. Scientific Discussion: Interpreting the Collapse & Variance")
    col_disc1, col_disc2 = st.columns(2)

    with col_disc1:
        st.markdown("""
        <div class="research-card">
            <div class="research-card-header">📉 Evidence of Generalization Collapse</div>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                1. <b>Systematic Performance Decay</b>: Across all five evaluation metrics (Accuracy, ROC-AUC, Precision, Recall, F1), the protein-disjoint split consistently underperforms the random split.
            </p>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                2. <b>Recall Drop is Most Severe</b>: Recall collapses by <b>-5.45%</b> (from 63.17% down to 57.72%). When faced with unseen proteins, the classifier struggles severely to recognize true positive interactions.
            </p>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                3. <b>Realistic Performance Bounds</b>: On global AAC features with linear classification, true out-of-distribution performance sits at ~58.1% accuracy and 0.615 ROC-AUC. Any study asserting 90%+ accuracy on such representations is unquestionably exploiting edge-level data leakage.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_disc2:
        st.markdown("""
        <div class="research-card">
            <div class="research-card-header">📊 Explaining the Standard Deviation Surge</div>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                <b>Why does the standard deviation explode by 4× to 5.6× in the Protein-Disjoint split?</b>
            </p>
            <ul style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                <li><b>Random Split Stability</b>: In random splitting, test edges are sampled uniformly across the entire graph. Every test set has identical size (3,050 pairs) and draws from the identical protein pool. Variance across seeds is artificially near-zero (<1%).</li>
                <li><b>Graph Topology Inhomogeneity</b>: Biological interactomes are scale-free networks. When 338 proteins are randomly chosen for the test partition, their internal edge count fluctuates drastically depending on whether hub proteins are partitioned into train or test.</li>
                <li><b>Uncertainty in Novel Evaluation</b>: Some test seeds contain well-represented protein families, while others consist of isolated, atypical proteins. Single-seed disjoint reporting is statistically invalid; 20-seed replication is imperative.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ======================================================================================
# PAGE 5: 20-SEED EXPERIMENT EXPLORER
# ======================================================================================
elif "5. 20-Seed Experiment" in selected_page:
    st.markdown("""
    <div class="hero-container">
        <span class="badge badge-emerald">Interactive Analytics</span>
        <span class="badge badge-blue">40 Replicates</span>
        <span class="badge badge-amber">Filterable Seeds</span>
        <div class="hero-title">20-Seed Experiment Explorer</div>
        <div class="hero-subtitle">
            Interactive examination of raw experiment records from twenty_seed_results.csv. Filter by split strategy, seed index, and analyze distribution trends.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_raw = load_csv_data("results/twenty_seed_results.csv")

    if df_raw is None:
        st.error("Error: results/twenty_seed_results.csv could not be found.")
    else:
        if "f1_score" in df_raw.columns and "f1" not in df_raw.columns:
            df_raw["f1"] = df_raw["f1_score"]

        col_f1, col_f2, col_f3 = st.columns([2, 3, 2])
        with col_f1:
            split_filter = st.selectbox(
                "Filter by Split Strategy:",
                ["All (Comparative View)", "Random Only", "Protein-disjoint Only"]
            )
        with col_f2:
            seed_range = st.slider(
                "Select Seed Range (0 to 19):",
                min_value=0,
                max_value=19,
                value=(0, 19)
            )
        with col_f3:
            metric_choice = st.selectbox(
                "Metric of Focus:",
                ["accuracy", "roc_auc", "f1", "recall", "precision"],
                format_func=lambda x: {
                    "accuracy": "Accuracy",
                    "roc_auc": "ROC-AUC",
                    "f1": "F1-Score",
                    "recall": "Recall",
                    "precision": "Precision"
                }[x]
            )

        filtered_df = df_raw[(df_raw["seed"] >= seed_range[0]) & (df_raw["seed"] <= seed_range[1])].copy()
        if split_filter == "Random Only":
            filtered_df = filtered_df[filtered_df["split"] == "Random"]
        elif split_filter == "Protein-disjoint Only":
            filtered_df = filtered_df[filtered_df["split"] == "Protein-disjoint"]

        st.markdown(f"#### 📊 Seed-by-Seed Comparison: {metric_choice.upper()}")
        
        fig_seed = px.bar(
            filtered_df,
            x="seed",
            y=metric_choice,
            color="split",
            barmode="group",
            color_discrete_map={"Random": COLOR_RANDOM, "Protein-disjoint": COLOR_DISJOINT},
            labels={"seed": "Seed Index", metric_choice: metric_choice.upper(), "split": "Split Strategy"},
            title=f"{metric_choice.upper()} Across Selected Seeds"
        )
        fig_seed.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1))
        fig_seed = apply_plotly_theme(fig_seed, title=f"{metric_choice.upper()} Across Seeds (Range: {seed_range[0]} - {seed_range[1]})", height=380)
        st.plotly_chart(fig_seed, use_container_width=True)

        col_box, col_scat = st.columns(2)
        with col_box:
            st.markdown(f"#### 📦 Variance Distribution ({metric_choice.upper()})")
            fig_box = px.box(
                filtered_df,
                x="split",
                y=metric_choice,
                color="split",
                points="all",
                color_discrete_map={"Random": COLOR_RANDOM, "Protein-disjoint": COLOR_DISJOINT},
                title="Spread and Outlier Seeds"
            )
            fig_box = apply_plotly_theme(fig_box, title=f"Distribution of {metric_choice.upper()}", height=340)
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_scat:
            st.markdown("#### 🔬 Test Set Size vs. Performance (Disjoint Split)")
            disjoint_only = df_raw[df_raw["split"] == "Protein-disjoint"]
            fig_scat = px.scatter(
                disjoint_only,
                x="test_pairs",
                y=metric_choice,
                hover_data=["seed", "accuracy", "roc_auc", "test_pairs"],
                labels={"test_pairs": "Number of Test Pairs", metric_choice: metric_choice.upper()},
                title="Impact of Variable Test Set Size"
            )
            fig_scat.update_traces(marker=dict(size=10, color=COLOR_DISJOINT, line=dict(width=1, color="#F8FAFC")))
            fig_scat = apply_plotly_theme(fig_scat, title=f"Test Pairs vs {metric_choice.upper()}", height=340)
            st.plotly_chart(fig_scat, use_container_width=True)

        st.markdown("### 🔎 Individual Seed Inspector")
        inspect_seed = st.selectbox("Inspect individual seed details:", list(range(20)), index=13)
        seed_data = df_raw[df_raw["seed"] == inspect_seed]

        c_rand = seed_data[seed_data["split"] == "Random"].iloc[0] if len(seed_data[seed_data["split"] == "Random"]) > 0 else None
        c_disj = seed_data[seed_data["split"] == "Protein-disjoint"].iloc[0] if len(seed_data[seed_data["split"] == "Protein-disjoint"]) > 0 else None

        if c_rand is not None and c_disj is not None:
            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            with col_s1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Seed {inspect_seed} Accuracy</div>
                    <div class="metric-value" style="font-size:1.3rem;">{c_disj['accuracy']*100:.2f}%</div>
                    <div class="metric-subtitle">Random: {c_rand['accuracy']*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Seed {inspect_seed} ROC-AUC</div>
                    <div class="metric-value" style="font-size:1.3rem;">{c_disj['roc_auc']:.4f}</div>
                    <div class="metric-subtitle">Random: {c_rand['roc_auc']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Seed {inspect_seed} Recall</div>
                    <div class="metric-value" style="font-size:1.3rem;">{c_disj['recall']*100:.2f}%</div>
                    <div class="metric-subtitle">Random: {c_rand['recall']*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Disjoint Test Pairs</div>
                    <div class="metric-value" style="font-size:1.3rem;">{c_disj['test_pairs']:,}</div>
                    <div class="metric-subtitle">Train: {c_disj['train_pairs']:,} pairs</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s5:
                delta_acc = (c_disj['accuracy'] - c_rand['accuracy']) * 100
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Collapse Delta (Δ)</div>
                    <div class="metric-value" style="font-size:1.3rem; color:{'#FB7185' if delta_acc < 0 else '#34D399'};">{delta_acc:+.2f}%</div>
                    <div class="metric-subtitle">Disjoint vs Random</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### 📋 Complete 20-Seed Data Records")
        st.dataframe(filtered_df, use_container_width=True)
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Results as CSV",
            data=csv_data,
            file_name="twenty_seed_results_filtered.csv",
            mime="text/csv"
        )

# ======================================================================================
# PAGE 6: CONCLUSIONS, LIMITATIONS & FUTURE WORK
# ======================================================================================
elif "6. Conclusions" in selected_page:
    st.markdown("""
    <div class="hero-container">
        <span class="badge badge-emerald">Research Synthesis</span>
        <span class="badge badge-amber">Critical Limitations</span>
        <span class="badge badge-blue">Future Research Directions</span>
        <div class="hero-title">Conclusions, Limitations & Future Work</div>
        <div class="hero-subtitle">
            Synthesizing experimental findings, rigorously stating biological limitations, and outlining next-generation research roadmaps.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1. Principal Experimental Findings")
    st.markdown("""
    <div class="research-card">
        <ol style="color:#CBD5E1; font-size:0.95rem; line-height:1.75; padding-left:22px;">
            <li><b>Empirical Validation of Generalization Collapse:</b> Our experiments conclusively demonstrate that evaluating PPI classifiers under random edge-level splits systematically overestimates model accuracy (mean drop: <b>61.39% → 58.13%</b>; ROC-AUC: <b>0.6619 → 0.6151</b>). Random splits mask the model's inability to predict interactions between uncharacterized proteins.</li>
            <li><b>Multi-Seed Variance is Crucial:</b> In strict protein-disjoint splits, variance increases nearly 4-fold (standard deviation jumps from <b>0.92% to 3.45%</b> in accuracy and from <b>0.0106 to 0.0426</b> in ROC-AUC). Single-seed evaluations in literature are susceptible to high variance and cherry-picking; multi-seed benchmarking (≥20 seeds) must be standard practice.</li>
            <li><b>Realistic Baseline Performance:</b> Linear classification on global Amino Acid Composition (AAC) provides an honest, leak-free baseline accuracy of ~58%. We explicitly refute uncorroborated claims of 90–95% accuracy on sequence AAC features as artifacts of edge-level memorization.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 2. Methodological & Biological Limitations")
    c_lim1, c_lim2, c_lim3 = st.columns(3)

    with c_lim1:
        st.markdown("""
        <div class="scientific-callout-rose" style="min-height:260px;">
            <b style="color:#FB7185; font-size:1.05rem;">1. Variable Disjoint Test Sizes</b>
            <p style="color:#CBD5E1; font-size:0.88rem; line-height:1.55; margin-top:8px;">
                Because biological protein networks are scale-free with heavy-tailed degree distributions, partitioning 20% of proteins produces test set sizes ranging between <b>368 and 860 pairs</b> across seeds.
                Seeds where test partitions lack high-degree hubs have fewer internal edges, leading to fluctuating statistical test power.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_lim2:
        st.markdown("""
        <div class="scientific-callout" style="min-height:260px;">
            <b style="color:#F59E0B; font-size:1.05rem;">2. Sampled Pseudo-Negatives</b>
            <p style="color:#CBD5E1; font-size:0.88rem; line-height:1.55; margin-top:8px;">
                Negative pairs were generated by uniform random pairing of proteins without recorded interactions in SHS27K.
                Because current interactome databases are inherently incomplete, <b>sampled negatives are pseudo-negatives</b>, not experimentally verified non-interacting pairs. Some unobserved pairs may be true interactions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_lim3:
        st.markdown(r"""
        <div class="scientific-callout-blue" style="min-height:260px;">
            <b style="color:#38BDF8; font-size:1.05rem;">3. Homology & Sequence Similarity</b>
            <p style="color:#CBD5E1; font-size:0.88rem; line-height:1.55; margin-top:8px;">
                Strict protein-disjoint splitting ensures $V_{train} \cap V_{test} = \emptyset$ (no protein ID overlap).
                However, <b>sequence similarity clustering was not conducted</b>. Homologous proteins (e.g. paralogs, shared kinase domains) across train and test partitions can still cause residual homology leakage.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 3. Future Research Directions (Prospective Roadmap)")
    st.markdown("""
    <div class="research-card">
        <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
            To advance beyond the foundational baseline established in this first-year research study, the following investigations are proposed as <b>future work</b>:
        </p>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:12px;">
            <div style="background-color:#080D1A; padding:14px 18px; border-radius:8px; border:1px solid #1E293B;">
                <b style="color:#38BDF8;">🔬 1. Sequence-Similarity-Disjoint Validation:</b><br>
                <span style="color:#CBD5E1; font-size:0.88rem; line-height:1.5;">
                Cluster protein sequences using MMseqs2 or CD-HIT at a 30% sequence identity threshold. Partition whole sequence clusters rather than individual protein IDs to completely eliminate homology leakage.
                </span>
            </div>
            <div style="background-color:#080D1A; padding:14px 18px; border-radius:8px; border:1px solid #1E293B;">
                <b style="color:#38BDF8;">🧠 2. Pretrained Protein Language Models (pLMs):</b><br>
                <span style="color:#CBD5E1; font-size:0.88rem; line-height:1.5;">
                Replace static AAC frequencies with contextual sequence embeddings from deep protein language models (e.g. ESM-2, ProtBERT) to evaluate whether evolutionary embeddings withstand generalization collapse.
                </span>
            </div>
            <div style="background-color:#080D1A; padding:14px 18px; border-radius:8px; border:1px solid #1E293B;">
                <b style="color:#38BDF8;">🕸️ 3. Inductive Graph Neural Networks (GNNs):</b><br>
                <span style="color:#CBD5E1; font-size:0.88rem; line-height:1.5;">
                Deploy inductive GNN architectures (e.g. GraphSAGE, Relational GCN) capable of predicting links on unseen subgraphs without requiring transductive node embedding access.
                </span>
            </div>
            <div style="background-color:#080D1A; padding:14px 18px; border-radius:8px; border:1px solid #1E293B;">
                <b style="color:#38BDF8;">🧪 4. Biologically-Constrained Negative Sampling:</b><br>
                <span style="color:#CBD5E1; font-size:0.88rem; line-height:1.5;">
                Incorporate cellular compartment annotations (e.g. nuclear vs. extracellular) to synthesize biologically grounded negative pairs that are physically segregated within the cell.
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 4. Academic Presentation & Citation")
    st.markdown("""
    <div style="background-color:#080D1A; border:1px solid #2A3B5C; border-radius:8px; padding:16px 20px;">
        <span style="color:#94A3B8; font-size:0.85rem; text-transform:uppercase; font-weight:600;">Recommended Citation / Presentation Reference</span>
        <pre style="color:#38BDF8; font-size:0.82rem; margin-top:8px; background:transparent; border:none; padding:0;">
@article{ppi_generalization_collapse_2026,
  title   = {Evaluating Generalization Collapse in ML-based PPI Prediction: A Comparative Study Across Random vs. Protein-Disjoint Data Splits},
  author  = {First-Year Undergraduate Research Group},
  year    = {2026},
  journal = {Academic Research Project Presentation},
  dataset = {SHS27K STRING Benchmark}
}
        </pre>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748B; font-size:0.82rem; padding:10px 0;">
    Evaluating Generalization Collapse in ML-based PPI Prediction &bull; Academic Research Dashboard &bull; Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
