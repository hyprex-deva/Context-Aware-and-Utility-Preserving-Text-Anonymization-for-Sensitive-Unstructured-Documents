"""
Streamlit Web Application: Context-Aware and Utility-Preserving Text Anonymization.

Features:
1. Two-Stage Anonymization Architecture (Stage 1 Synthetic Surrogates + Stage 2 Semantic SLM).
2. 3-Column Visual Comparison Layout with Entity Highlighting.
3. Microsoft Presidio & Rigid Redaction Baseline Comparison Drawer.
4. Semantic Drift Guardrail (SentenceTransformers all-MiniLM-L6-v2) Audit Panel.
"""

from __future__ import annotations

import html
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.stage1_direct_pii import DirectPIIAnonymizer
from src.stage2_semantic_defense import SemanticQuasiIdentifierDefense
from src.baselines import BaselinePresidio, BaselineRedacted
from src.evaluate_metrics import compute_bleu_scores, compute_rouge_l_scores

# Configure Streamlit page
st.set_page_config(
    page_title="Context-Aware Text Anonymizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }
    
    .card-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        min-height: 260px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        line-height: 1.6;
        font-size: 14.5px;
    }
    
    .card-header {
        font-weight: 600;
        font-size: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .badge-per { background-color: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-org { background-color: #e0e7ff; color: #3730a3; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-loc { background-color: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-email { background-color: #d1fae5; color: #065f46; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-phone { background-color: #fce7f3; color: #831843; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-ssn { background-color: #ede9fe; color: #5b21b6; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-ip { background-color: #ccfbf1; color: #115e59; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-card { background-color: #ffedd5; color: #9a3412; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-id { background-color: #f3e8ff; color: #6b21a8; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-license { background-color: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-pass { background-color: #fef2f2; color: #b91c1c; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }
    .badge-qi { background-color: #e2e8f0; color: #334155; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 13px; }

    
    .guardrail-pass {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .guardrail-fallback {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_stage1_anonymizer() -> DirectPIIAnonymizer:
    return DirectPIIAnonymizer(use_ner=True)


@st.cache_resource
def get_stage2_defense(
    tau: float = 0.72,
    floor_sim: float = 0.60,
    w_sim: float = 0.50,
    w_priv: float = 0.30,
    w_read: float = 0.20,
) -> SemanticQuasiIdentifierDefense:
    return SemanticQuasiIdentifierDefense(
        ollama_model="qwen2.5:1.5b",
        tau=tau,
        floor_sim=floor_sim,
        weights={"sim": w_sim, "priv": w_priv, "read": w_read},
    )


@st.cache_resource
def get_presidio_baseline() -> BaselinePresidio:
    return BaselinePresidio()


@st.cache_resource
def get_redacted_baseline() -> BaselineRedacted:
    s1 = get_stage1_anonymizer()
    return BaselineRedacted(stage1_anonymizer=s1)


@st.cache_resource
def load_full_dataset():
    """Load the complete ai4privacy/pii-masking-300k dataset cached locally."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4privacy/pii-masking-300k", split="train")
        return ds
    except Exception as e:
        logger.warning(f"Error loading full dataset: {e}")
        return None


def get_dataset_sample(ds, idx: int) -> Dict[str, Any]:
    """Retrieve sample text and metadata by index."""
    if ds is not None and 0 <= idx < len(ds):
        row = ds[idx]
        masks = row.get("privacy_mask", [])
        direct_pii = [
            {"entity": m["value"], "type": m["label"], "start": m["start"], "end": m["end"]}
            for m in masks
        ]
        return {
            "id": row.get("id", f"sample_{idx}"),
            "text": row.get("source_text", ""),
            "target_text": row.get("target_text", ""),
            "direct_pii": direct_pii,
            "language": row.get("language", "English"),
        }
    return {
        "id": "fallback_01",
        "text": "Please enter your unstructured text here to sanitize.",
        "target_text": "",
        "direct_pii": [],
        "language": "English",
    }


def highlight_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    """Generate color-coded HTML highlighting for detected entities."""
    if not entities or not text:
        return html.escape(text)

    # Sort entities by start offset descending
    sorted_ents = sorted(entities, key=lambda x: x.get("start", 0), reverse=True)
    chars = list(text)

    tag_classes = {
        "PER": "badge-per",
        "PERSON": "badge-per",
        "ORG": "badge-org",
        "ORGANIZATION": "badge-org",
        "LOC": "badge-loc",
        "LOCATION": "badge-loc",
        "GPE": "badge-loc",
        "EMAIL": "badge-email",
        "PHONE": "badge-phone",
        "SSN": "badge-ssn",
        "IP_ADDRESS": "badge-ip",
        "CREDIT_CARD": "badge-card",
        "ID_CARD": "badge-id",
        "IDCARD": "badge-id",
        "LICENSE": "badge-license",
        "DRIVER_LICENSE": "badge-license",
        "PASSWORD": "badge-pass",
        "MISC": "badge-qi",
    }

    for ent in sorted_ents:
        start = ent.get("start", 0)
        end = ent.get("end", 0)
        etype = ent.get("entity_type", "MISC").upper()
        css_cls = tag_classes.get(etype, "badge-qi")
        val = html.escape(text[start:end])
        badge_html = f"<span class='{css_cls}' title='Type: {etype}'>{val} <small style='opacity:0.7;'>({etype})</small></span>"
        chars[start:end] = list(badge_html)

    return "".join(chars)


def main():
    # Application Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🛡️ Context-Aware & Utility-Preserving Text Anonymization</h1>
            <p>
                Two-Stage Architecture: <strong>Stage 1</strong> (Deterministic Synthetic Surrogates via Transformer NER + Regex) 
                &rarr; <strong>Stage 2</strong> (Hierarchical Quasi-Identifier Generalization via <strong>Ollama qwen2.5:1.5b</strong> + Multi-Criteria Composite Decision Guardrail).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    full_ds = load_full_dataset()
    total_samples = len(full_ds) if full_ds is not None else 0

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration & Controls")
        
        st.subheader("📚 Dataset: `ai4privacy/pii-masking-300k`")
        if total_samples > 0:
            st.caption(f"Loaded **{total_samples:,}** total benchmark documents.")
            
            sample_idx = st.number_input(
                "🔢 Choose Sample Index (0 to total)",
                min_value=0,
                max_value=total_samples - 1,
                value=0,
                step=1,
                help=f"Select any document index from 0 to {total_samples-1}.",
            )
            current_sample = get_dataset_sample(full_ds, sample_idx)
            st.info(f"**Document ID:** `{current_sample['id']}` ({current_sample['language']})")
        else:
            st.warning("Hugging Face dataset not found locally. Loading defaults.")
            current_sample = {
                "id": "sample_0",
                "text": "License: MINDA.658073.MR.352\n- IP Address: 1dca:680f:2938:6035:4ed8:81d:c6d6:3b1a\n- Password: \"{0w7/U\n\nOther Candidates:\n- Candidate C: Email: asukas55@aol.com, ID Card Number: UK57900JK\n- Candidate D: Email: 3chunmei@protonmail.com, ID Card Number: UGG576437H\n- Candidate E: Email: ydtjqhxrfiv1162@hotmail.com, ID Card Number: OU79828NR\n- Candidate F: Email: A@protonmail.com, ID Card Number: FE15976DV\n- Candidate G: Email: tdjispgtfiqx547@tutanot",
                "direct_pii": [],
                "language": "English",
            }

        st.divider()
        st.subheader("Stage 2 Decision Engine")
        st.success("🦙 **Ollama SLM**: `qwen2.5:1.5b` (Active)")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tau_threshold = st.slider(
                "Composite Threshold (τ)",
                min_value=0.50,
                max_value=0.95,
                value=0.72,
                step=0.02,
                help="Minimum composite score (S_composite) required to accept candidate rewrite.",
            )
        with col_t2:
            floor_sim = st.slider(
                "Safety Sim Floor",
                min_value=0.40,
                max_value=0.85,
                value=0.60,
                step=0.05,
                help="Hard semantic similarity floor (S_semantic) required regardless of composite score.",
            )

        with st.expander("⚖️ Decision Weights Tuning", expanded=False):
            w_sim = st.slider("Semantic Weight (w_sim)", 0.0, 1.0, 0.50, 0.05)
            w_priv = st.slider("Privacy Weight (w_priv)", 0.0, 1.0, 0.30, 0.05)
            w_read = st.slider("Readability Weight (w_read)", 0.0, 1.0, 0.20, 0.05)

        st.divider()
        st.markdown(
            """
            **Legend:**
            - <span class='badge-per'>PER</span> Person Names
            - <span class='badge-org'>ORG</span> Organizations
            - <span class='badge-loc'>LOC</span> Locations / Addresses
            - <span class='badge-email'>EMAIL</span> Email Addresses
            - <span class='badge-phone'>PHONE</span> Phone Numbers
            - <span class='badge-ssn'>SSN</span> Social Security Numbers
            - <span class='badge-ip'>IP</span> IP Addresses (v4/v6)
            - <span class='badge-card'>CARD</span> Credit Cards
            - <span class='badge-id'>ID_CARD</span> ID Card / National IDs
            - <span class='badge-license'>LICENSE</span> Driver / State Licenses
            - <span class='badge-pass'>PASSWORD</span> Passwords / Secrets
            """,
            unsafe_allow_html=True,
        )

    # Text Input Area
    input_text = st.text_area(
        "📝 Input Unstructured Document",
        value=current_sample["text"],
        height=130,
        help="Paste unstructured document containing direct PII and subtle quasi-identifiers.",
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        sanitize_btn = st.button("🚀 Sanitize Document", type="primary", use_container_width=True)

    if sanitize_btn or input_text:
        # Load pipelines
        with st.spinner("Processing through Two-Stage Anonymization Architecture (Ollama qwen2.5:1.5b)..."):
            stage1_engine = get_stage1_anonymizer()
            stage2_engine = get_stage2_defense(
                tau=tau_threshold,
                floor_sim=floor_sim,
                w_sim=w_sim,
                w_priv=w_priv,
                w_read=w_read,
            )
            presidio_engine = get_presidio_baseline()
            redacted_engine = get_redacted_baseline()

            # Execute Stage 1
            s1_result = stage1_engine.anonymize(input_text)
            
            # Execute Stage 2 with Composite Engine
            s2_result = stage2_engine.generalize_text(
                stage1_text=s1_result["sanitized_text"],
                original_text=input_text,
                tau=tau_threshold,
                floor_sim=floor_sim,
                weights={"sim": w_sim, "priv": w_priv, "read": w_read},
            )

            # Execute Baselines
            presidio_res = presidio_engine.anonymize(input_text)
            redacted_res = redacted_engine.anonymize(input_text)

        # 3-Column Comparative Layout
        st.markdown("### 🔍 Comparative Anonymization Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="card-box">
                    <div class="card-header" style="color: #0f172a;">
                        <span>📄 1. Original Document</span>
                    </div>
                    <div style="line-height: 1.8;">
                        {highlight_entities(input_text, s1_result["detected_entities"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="card-box">
                    <div class="card-header" style="color: #2563eb;">
                        <span>🎭 2. Stage 1: Synthetic Surrogates</span>
                    </div>
                    <div style="line-height: 1.8; color: #1e293b;">
                        {html.escape(s1_result["sanitized_text"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            s_comp = s2_result.get("composite_score", 0.0)
            metrics_bd = s2_result.get("metrics_breakdown", {})
            s_sim = metrics_bd.get("semantic_similarity", s2_result.get("similarity_score", 0.0))
            is_acc = s2_result.get("is_accepted", s2_result.get("drift_passed", True))

            guardrail_badge = (
                f"<div class='guardrail-pass'>✅ Guardrail Accepted (S<sub>comp</sub>: {s_comp:.2f} &ge; {tau_threshold:.2f} | S<sub>sim</sub>: {s_sim:.2f} &ge; {floor_sim:.2f})</div>"
                if is_acc
                else f"<div class='guardrail-fallback'>⚠️ Fallback Triggered (S<sub>comp</sub>: {s_comp:.2f} | Floor: {floor_sim:.2f})</div>"
            )

            st.markdown(
                f"""
                <div class="card-box">
                    <div class="card-header" style="color: #059669; justify-content: space-between;">
                        <span>🛡️ 3. Stage 2: Contextual Defense</span>
                    </div>
                    <div style="margin-bottom: 12px;">{guardrail_badge}</div>
                    <div style="line-height: 1.8; color: #1e293b;">
                        {html.escape(s2_result["final_text"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Baseline Comparison Drawer
        with st.expander("🏢 Enterprise & Rigid Baselines Comparison (Presidio vs. Redaction)", expanded=False):
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                st.markdown("**Microsoft Presidio (<TAGS>)**")
                st.info(presidio_res["sanitized_text"])
                r_pres = compute_rouge_l_scores([input_text], [presidio_res["sanitized_text"]])["rougeL_f1"]
                st.caption(f"ROUGE-L Retention: **{r_pres:.4f}**")
            with bcol2:
                st.markdown("**Rigid Redaction ([REDACTED])**")
                st.warning(redacted_res["sanitized_text"])
                r_red = compute_rouge_l_scores([input_text], [redacted_res["sanitized_text"]])["rougeL_f1"]
                st.caption(f"ROUGE-L Retention: **{r_red:.4f}**")

        # Privacy & Utility Audit Panel
        with st.expander("📊 Privacy & Utility Audit Dashboard", expanded=True):
            tab1, tab2, tab3 = st.tabs(["🏷️ Detected Entities & Surrogates", "🧠 Semantic Generalization & Guardrail Logs", "📈 Quantitative Utility Metrics"])

            with tab1:
                if s1_result["detected_entities"]:
                    df_entities = pd.DataFrame([
                        {
                            "Raw Entity Span": e["entity_value"],
                            "Type": e["entity_type"],
                            "Detector": e["detector"],
                            "Synthetic Surrogate": e["surrogate_value"],
                            "Start Offset": e["start"],
                            "End Offset": e["end"],
                            "Confidence": e["score"],
                        }
                        for e in s1_result["detected_entities"]
                    ])
                    st.dataframe(df_entities, use_container_width=True)
                else:
                    st.write("No direct PII entities detected in this document.")

            with tab2:
                # Composite Decision Telemetry
                st.markdown("#### 🎯 Composite Guardrail Telemetry")
                gcol1, gcol2, gcol3, gcol4 = st.columns(4)
                with gcol1:
                    st.metric("Composite Score (S_comp)", f"{s2_result.get('composite_score', 0.0):.4f}", f"Threshold τ = {tau_threshold:.2f}")
                with gcol2:
                    st.metric("Semantic Sim (S_sim)", f"{s2_result.get('metrics_breakdown', {}).get('semantic_similarity', 0.0):.4f}", f"Floor = {floor_sim:.2f}")
                with gcol3:
                    st.metric("Privacy Reduction (S_priv)", f"{s2_result.get('metrics_breakdown', {}).get('privacy_reduction_score', 1.0):.4f}")
                with gcol4:
                    st.metric("Readability (S_read)", f"{s2_result.get('metrics_breakdown', {}).get('readability_score', 1.0):.4f}")

                st.divider()
                mods = s2_result.get("modifications") or s2_result.get("generalized_spans", [])
                if mods:
                    st.markdown(f"**Flagged Quasi-Identifiers & Applied Generalizations:** ({len(mods)} spans)")
                    df_mods = pd.DataFrame(mods)
                    st.dataframe(df_mods, use_container_width=True)
                    
                    st.markdown("**Candidate Generalized Text (Before Decision Guardrail):**")
                    st.code(s2_result.get("candidate_text", ""), language="text")
                    
                    if not is_acc:
                        st.warning(
                            f"⚠️ Multi-Criteria Guardrail Fallback Triggered: Candidate score failed acceptance criteria "
                            f"(Composite: {s_comp:.4f} vs threshold {tau_threshold:.2f}, Semantic: {s_sim:.4f} vs floor {floor_sim:.2f}). "
                            f"Stage 1 surrogate text was safely retained in Column 3 to guarantee document utility."
                        )
                else:
                    st.info("No high-risk quasi-identifiers flagged by the reasoning SLM for this document.")

            with tab3:
                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                r_s1 = compute_rouge_l_scores([input_text], [s1_result["sanitized_text"]])["rougeL_f1"]
                r_s2 = compute_rouge_l_scores([input_text], [s2_result["final_text"]])["rougeL_f1"]
                b_s1 = compute_bleu_scores([input_text], [s1_result["sanitized_text"]])["bleu_score"]
                b_s2 = compute_bleu_scores([input_text], [s2_result["final_text"]])["bleu_score"]

                with m_col1:
                    st.metric("Direct Entities Masked", len(s1_result["detected_entities"]))
                with m_col2:
                    st.metric("Stage 1 ROUGE-L", f"{r_s1:.4f}")
                with m_col3:
                    st.metric("Stage 2 ROUGE-L", f"{r_s2:.4f}")
                with m_col4:
                    st.metric("Stage 2 BLEU-4", f"{b_s2:.4f}")
                with m_col5:
                    st.metric("Composite Score", f"{s2_result.get('composite_score', 0.0):.4f}")


if __name__ == "__main__":
    main()
