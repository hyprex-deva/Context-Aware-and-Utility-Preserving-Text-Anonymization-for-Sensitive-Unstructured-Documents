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
from src.evaluate_metrics import compute_rouge_l_scores

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
    ollama_model: str = "qwen2.5:1.5b", threshold: float = 0.80
) -> SemanticQuasiIdentifierDefense:
    return SemanticQuasiIdentifierDefense(
        ollama_model=ollama_model,
        similarity_threshold=threshold,
    )





@st.cache_resource
def get_presidio_baseline() -> BaselinePresidio:
    return BaselinePresidio()


@st.cache_resource
def get_redacted_baseline() -> BaselineRedacted:
    s1 = get_stage1_anonymizer()
    return BaselineRedacted(stage1_anonymizer=s1)


def load_preset_examples() -> List[Dict[str, Any]]:
    benchmark_path = os.path.join(os.path.dirname(__file__), "data", "benchmark_samples.json")
    if os.path.exists(benchmark_path):
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return [
        {
            "id": "sample_01",
            "domain": "Healthcare",
            "text": "Patient Eleanor Vance (DOB: 12/04/1982, SSN: 482-19-8921) visited Dr. Robert Langdon at Johns Hopkins Hospital. She can be reached at eleanor.vance@medmail.org or (410) 555-0199. Eleanor works as the sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors in Baltimore, Maryland.",
            "direct_pii": [],
            "quasi_identifiers": [],
        }
    ]


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
                &rarr; <strong>Stage 2</strong> (Hierarchical Quasi-Identifier Generalization via SLM + MiniLM Semantic Drift Guardrail).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preset_samples = load_preset_examples()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration & Controls")
        
        # Domain template selector
        sample_options = [f"{s['domain']}: {s['id']}" for s in preset_samples]
        selected_idx = st.selectbox(
            "📁 Load Domain Template",
            range(len(sample_options)),
            format_func=lambda i: sample_options[i],
            index=0,
        )
        current_sample = preset_samples[selected_idx]

        st.divider()
        st.subheader("Stage 2 Reasoning Engine")
        ollama_model_choice = st.selectbox(
            "🦙 Ollama Model",
            ["qwen2.5:1.5b", "qwen2.5:0.5b"],
            index=0,
            help="qwen2.5:1.5b has strong instruction-following reasoning to generalize dates/amounts/roles. qwen2.5:0.5b is ultra-lightweight.",
        )
        
        sim_threshold = st.slider(
            "Semantic Drift Threshold (Cosine Sim)",
            min_value=0.50,
            max_value=0.95,
            value=0.80,
            step=0.05,
            help="Minimum cosine similarity required by MiniLM guardrail to accept generalized text.",
        )

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
        with st.spinner(f"Processing through Two-Stage Anonymization Architecture (Ollama {ollama_model_choice})..."):
            stage1_engine = get_stage1_anonymizer()
            stage2_engine = get_stage2_defense(
                ollama_model=ollama_model_choice,
                threshold=sim_threshold,
            )

            presidio_engine = get_presidio_baseline()
            redacted_engine = get_redacted_baseline()

            # Execute Stage 1
            s1_result = stage1_engine.anonymize(input_text)
            
            # Execute Stage 2
            s2_result = stage2_engine.generalize_text(
                stage1_text=s1_result["sanitized_text"], original_text=input_text
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
            guardrail_badge = (
                f"<div class='guardrail-pass'>✅ Drift Guardrail Passed ({s2_result['similarity_score']:.2f} &ge; {sim_threshold})</div>"
                if s2_result["drift_passed"]
                else f"<div class='guardrail-fallback'>⚠️ Drift Fallback Triggered ({s2_result['similarity_score']:.2f} &lt; {sim_threshold})</div>"
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
            tab1, tab2, tab3 = st.tabs(["🏷️ Detected Entities & Surrogates", "🧠 Semantic Generalization Logs", "📈 Quantitative Utility Metrics"])

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
                if s2_result["generalized_spans"]:
                    st.markdown(f"**Applied Contextual Transformations & Rephrasing:** ({len(s2_result['generalized_spans'])} spans)")
                    df_mods = pd.DataFrame(s2_result["generalized_spans"])
                    st.dataframe(df_mods, use_container_width=True)
                    
                    st.markdown("**Candidate Generalized Text (Before Guardrail):**")
                    st.code(s2_result["candidate_text"], language="text")
                    
                    if not s2_result["drift_passed"]:
                        st.warning(
                            f"⚠️ Semantic Drift Guardrail Triggered: The candidate text has a cosine similarity of "
                            f"{s2_result['similarity_score']:.4f} (below the threshold of {sim_threshold:.2f}). "
                            f"Stage 1 surrogate text was retained in Column 3 to guarantee document utility safety."
                        )
                else:
                    st.info("No high-risk quasi-identifiers or unique role patterns matched in this document.")


            with tab3:
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                r_s1 = compute_rouge_l_scores([input_text], [s1_result["sanitized_text"]])["rougeL_f1"]
                r_s2 = compute_rouge_l_scores([input_text], [s2_result["final_text"]])["rougeL_f1"]

                with m_col1:
                    st.metric("Direct Entities Masked", len(s1_result["detected_entities"]))
                with m_col2:
                    st.metric("Stage 1 ROUGE-L", f"{r_s1:.4f}")
                with m_col3:
                    st.metric("Stage 2 ROUGE-L", f"{r_s2:.4f}")
                with m_col4:
                    st.metric("MiniLM Cosine Similarity", f"{s2_result['similarity_score']:.4f}")


if __name__ == "__main__":
    main()
