"""
Generate Publication-Quality Visualizations for Text Anonymization Evaluation Benchmark
Corpus: AI4Privacy PII-Masking-300K (100 Stratified Benchmark Documents)

Figures Generated:
1. fig_eval_system_comparison.png - Privacy vs. Utility Benchmark Matrix & Pareto Frontier
2. fig_eval_entity_breakdown.png   - Direct PII Extraction Performance by Entity Type
3. fig_eval_qi_mitigation.png     - Stage 2 Quasi-Identifier Abstraction & Risk Mitigation
4. fig_eval_guardrail_telemetry.png - Multi-Criteria Composite Guardrail Dynamics & Ablation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Set global publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Helvetica', 'Arial'],
    'axes.edgecolor': '#94A3B8',
    'axes.linewidth': 1.1,
    'grid.color': '#E2E8F0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.7,
    'figure.titlesize': 15,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.autolayout': False
})

OUTPUT_DIR = os.path.join(os.getcwd(), 'Paper', 'Figures')
ARTIFACT_DIR = r"C:\Users\KIIT\.gemini\antigravity-ide\brain\e8c08ea8-65f1-4997-8006-30b34322939b"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ==============================================================================
# FIGURE 1: Comprehensive System Comparison Matrix (Privacy vs. Utility Tradeoff)
# ==============================================================================
def plot_figure_1(df_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8), gridspec_kw={'width_ratios': [1.3, 1.0]})
    
    systems = df_results['System'].tolist()
    # Shorter labels for clean plotting
    sys_labels = [
        'Rigid Redaction\n[REDACTED]',
        'Microsoft Presidio\n<TAGS>',
        'Proposed: Stage 1\n(Surrogates)',
        'Proposed: Two-Stage\n(Surrogates + QI)'
    ]
    
    # 1A: Grouped Bar Chart of Core Metrics
    metrics = ['PII F1 (Relaxed)', 'PII Removal Rate', 'Semantic Cosine Sim', 'BERTScore F1', 'Readability Score']
    metric_labels = ['PII F1\n(Relaxed)', 'PII Removal\nRate', 'Semantic\nCosine Sim', 'BERTScore\nF1', 'Readability\nScore']
    
    x = np.arange(len(metric_labels))
    width = 0.19
    
    colors = ['#EF4444', '#F59E0B', '#3B82F6', '#10B981']
    
    for i, sys in enumerate(systems):
        vals = [float(df_results.loc[df_results['System'] == sys, m].values[0]) for m in metrics]
        rects = ax1.bar(x + (i - 1.5) * width, vals, width, label=sys_labels[i], color=colors[i], alpha=0.92, edgecolor='#1E293B', linewidth=0.8)
        # Add values on top of bars
        for rect in rects:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2., h + 0.012, f'{h:.2f}',
                     ha='center', va='bottom', fontsize=8, rotation=0, fontweight='semibold', color='#1E293B')

    ax1.set_title('A. Cross-System Metric Performance Matrix', fontweight='bold', pad=12, loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontweight='medium')
    ax1.set_ylabel('Score (Normalized [0.0 - 1.0])', fontweight='bold')
    ax1.set_ylim(0.0, 1.12)
    ax1.grid(True, axis='y')
    ax1.legend(loc='lower left', framealpha=0.95, edgecolor='#CBD5E1', ncol=2, fontsize=9)
    
    # 1B: Pareto Tradeoff (Privacy Removal Rate vs. Downstream Semantic BERTScore)
    ax2.set_title('B. Privacy-Utility Pareto Space', fontweight='bold', pad=12, loc='left')
    
    markers = ['s', '^', 'D', 'o']
    for i, sys in enumerate(systems):
        rem_rate = float(df_results.loc[df_results['System'] == sys, 'PII Removal Rate'].values[0])
        bert = float(df_results.loc[df_results['System'] == sys, 'BERTScore F1'].values[0])
        cos = float(df_results.loc[df_results['System'] == sys, 'Semantic Cosine Sim'].values[0])
        read = float(df_results.loc[df_results['System'] == sys, 'Readability Score'].values[0])
        
        sc = ax2.scatter(rem_rate, bert, color=colors[i], s=260, marker=markers[i],
                         edgecolor='#0F172A', linewidth=1.5, zorder=5, label=sys_labels[i])
        
        # Offset annotation
        offsets = {0: (-15, -25), 1: (15, -15), 2: (15, 12), 3: (-120, 15)}
        ox, oy = offsets[i]
        ax2.annotate(
            f"{sys.replace('Proposed: ', '')}\nBERT: {bert:.3f} | Rem: {rem_rate:.3f}",
            (rem_rate, bert),
            xytext=(ox, oy), textcoords='offset points',
            fontsize=9, fontweight='bold', color=colors[i],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8FAFC', edgecolor=colors[i], alpha=0.85),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15', color=colors[i], lw=1.2)
        )
    
    # Pareto frontier line between Stage 1 and Two-Stage
    p1 = (float(df_results.loc[df_results['System'] == 'Proposed: Stage 1 (Surrogates)', 'PII Removal Rate'].values[0]),
          float(df_results.loc[df_results['System'] == 'Proposed: Stage 1 (Surrogates)', 'BERTScore F1'].values[0]))
    p2 = (float(df_results.loc[df_results['System'] == 'Proposed: Two-Stage (Stage 1 + Stage 2)', 'PII Removal Rate'].values[0]),
          float(df_results.loc[df_results['System'] == 'Proposed: Two-Stage (Stage 1 + Stage 2)', 'BERTScore F1'].values[0]))
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#10B981', linestyle=':', linewidth=2, label='Proposed Pareto Optimal Frontier')
    
    ax2.set_xlabel('Privacy: PII Removal Rate $\\uparrow$', fontweight='bold')
    ax2.set_ylabel('Utility: BERTScore F1 $\\uparrow$', fontweight='bold')
    ax2.set_xlim(0.48, 0.58)
    ax2.set_ylim(0.83, 0.94)
    ax2.grid(True)
    ax2.legend(loc='lower left', framealpha=0.95, edgecolor='#CBD5E1', fontsize=8.5)
    
    plt.suptitle('Figure 1: Benchmark Comparison Across Privacy & Downstream Semantic Utility (N=100 AI4Privacy)',
                 fontsize=14, fontweight='bold', y=0.98, color='#0F172A')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = os.path.join(OUTPUT_DIR, 'fig_eval_system_comparison.png')
    art_path = os.path.join(ARTIFACT_DIR, 'fig_eval_system_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(art_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 2: Direct PII Detection & Extraction Performance by Entity Category
# ==============================================================================
def plot_figure_2(df_entity):
    # Filter active entities with true positives > 0
    active = df_entity[df_entity['true_positives'] > 0].copy()
    active = active.sort_values(by='f1', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    y = np.arange(len(active))
    height = 0.26
    
    c_p = '#3B82F6'   # Precision: Blue
    c_r = '#10B981'   # Recall: Green
    c_f1 = '#6366F1'  # F1: Indigo
    
    rects1 = ax.barh(y + height, active['precision'], height, label='Precision', color=c_p, alpha=0.88, edgecolor='#1E293B')
    rects2 = ax.barh(y, active['recall'], height, label='Recall', color=c_r, alpha=0.88, edgecolor='#1E293B')
    rects3 = ax.barh(y - height, active['f1'], height, label='F1-Score', color=c_f1, alpha=0.95, edgecolor='#1E293B')
    
    # Add numerical labels inside/outside bars
    for rect in rects3:
        w = rect.get_width()
        ax.text(w + 0.015, rect.get_y() + rect.get_height() / 2., f'{w:.3f}',
                ha='left', va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
        
    # Annotate support count
    for idx, (i, row) in enumerate(active.iterrows()):
        tp = int(row['true_positives'])
        fp = int(row['false_positives'])
        fn = int(row['false_negatives'])
        ax.text(0.02, y[idx] - height, f"TP={tp} | FP={fp} | FN={fn}",
                ha='left', va='center', fontsize=7.5, color='#FFFFFF', fontweight='bold')
    
    ax.set_yticks(y)
    ax.set_yticklabels(active['entity_type'], fontweight='bold', fontsize=10)
    ax.set_xlabel('Score (Precision / Recall / F1)', fontweight='bold')
    ax.set_xlim(0.0, 1.15)
    ax.grid(True, axis='x')
    
    # Categorization bands
    ax.axvspan(0.90, 1.05, color='#10B981', alpha=0.06, label='High Accuracy Tier (F1 $\\geq$ 0.90)')
    
    ax.set_title('Figure 2: Stage 1 Direct PII Extraction Performance by Entity Category (Regex + Transformer NER)',
                 fontweight='bold', pad=15, fontsize=13, color='#0F172A', loc='left')
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), framealpha=0.95, edgecolor='#CBD5E1', fontsize=9.5, ncol=4)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    out_path = os.path.join(OUTPUT_DIR, 'fig_eval_entity_breakdown.png')
    art_path = os.path.join(ARTIFACT_DIR, 'fig_eval_entity_breakdown.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(art_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 3: Stage 2 Quasi-Identifier (QI) Abstraction & Risk Mitigation Analysis
# ==============================================================================
def plot_figure_3(df_qi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={'width_ratios': [1.0, 1.3]})
    
    # Categories: DATE, DEMOGRAPHIC, MONEY
    categories = df_qi['QI Type'].tolist()
    counts = df_qi['Count'].tolist()
    cat_labels = [f"{cat}\n(n={cnt})" for cat, cnt in zip(categories, counts)]
    
    # 3A: Risk Reduction (Initial vs. Residual Risk)
    x = np.arange(len(categories))
    w = 0.32
    
    init_risk = df_qi['Average Initial Risk'].tolist()
    res_risk = df_qi['Average Residual Risk'].tolist()
    
    rects_init = ax1.bar(x - w/2, init_risk, w, label='Initial Re-ID Risk', color='#EF4444', alpha=0.9, edgecolor='#1E293B')
    rects_res = ax1.bar(x + w/2, res_risk, w, label='Residual Re-ID Risk (After Stage 2)', color='#10B981', alpha=0.9, edgecolor='#1E293B')
    
    for r1, r2 in zip(rects_init, rects_res):
        h1 = r1.get_height()
        h2 = r2.get_height()
        ax1.text(r1.get_x() + r1.get_width()/2., h1 + 0.02, f"{h1:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(r2.get_x() + r2.get_width()/2., h2 + 0.02, f"{h2:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#047857')
        
        # Risk reduction delta
        reduction = (1.0 - h2) * 100
        if reduction > 0:
            ax1.annotate(f"-{reduction:.1f}%",
                         xy=(r2.get_x() + r2.get_width()/2., h2/2),
                         ha='center', va='center', fontsize=9, fontweight='bold', color='#FFFFFF',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#065F46', alpha=0.9, edgecolor='none'))

    ax1.set_title('A. Average Re-Identification Risk Reduction', fontweight='bold', pad=12, loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels, fontweight='bold')
    ax1.set_ylabel('Risk Level [0.0 = Safe, 1.0 = Max Linkage]', fontweight='bold')
    ax1.set_ylim(0.0, 1.22)
    ax1.grid(True, axis='y')
    ax1.legend(loc='upper right', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9)
    
    # 3B: Proportional Breakdown (Mitigated, Partial, Exposed)
    mitigated = np.array(df_qi['Mitigated Count'].tolist())
    partial = np.array(df_qi['Partial Count'].tolist())
    exposed = np.array(df_qi['Exposed Count'].tolist())
    total = np.array(counts)
    
    pct_mit = (mitigated / total) * 100
    pct_part = (partial / total) * 100
    pct_exp = (exposed / total) * 100
    
    y = np.arange(len(categories))
    bar_h = 0.45
    
    p1 = ax2.barh(y, pct_mit, bar_h, label=r'Fully Mitigated ($r_i \leq 0.10$)', color='#10B981', edgecolor='#1E293B', alpha=0.95)
    p2 = ax2.barh(y, pct_part, bar_h, left=pct_mit, label=r'Partially Coarsened ($0.10 < r_i < 0.90$)', color='#F59E0B', edgecolor='#1E293B', alpha=0.95)
    p3 = ax2.barh(y, pct_exp, bar_h, left=pct_mit + pct_part, label=r'Exposed / Unmodified ($r_i \geq 0.90$)', color='#EF4444', edgecolor='#1E293B', alpha=0.95)
    
    # Add percentage labels
    for i in range(len(categories)):
        # Mitigated label
        if pct_mit[i] > 8:
            ax2.text(pct_mit[i] / 2, y[i], f"{pct_mit[i]:.1f}%\n({mitigated[i]})",
                     ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=8.5)
        # Partial label
        if pct_part[i] > 8:
            ax2.text(pct_mit[i] + pct_part[i] / 2, y[i], f"{pct_part[i]:.1f}%\n({partial[i]})",
                     ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=8.5)
        # Exposed label
        if pct_exp[i] > 8:
            ax2.text(pct_mit[i] + pct_part[i] + pct_exp[i] / 2, y[i], f"{pct_exp[i]:.1f}%\n({exposed[i]})",
                     ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=8.5)
            
    ax2.set_title('B. Quasi-Identifier Mitigation Status Proportions', fontweight='bold', pad=12, loc='left')
    ax2.set_yticks(y)
    ax2.set_yticklabels(cat_labels, fontweight='bold')
    ax2.set_xlabel('Percentage of Detected Quasi-Identifiers (%)', fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.grid(True, axis='x')
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), framealpha=0.95, edgecolor='#CBD5E1', ncol=3, fontsize=9)
    
    plt.suptitle('Figure 3: Stage 2 SLM Quasi-Identifier (QI) Generalization & Residual Risk Analysis',
                 fontsize=14, fontweight='bold', y=0.98, color='#0F172A')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    out_path = os.path.join(OUTPUT_DIR, 'fig_eval_qi_mitigation.png')
    art_path = os.path.join(ARTIFACT_DIR, 'fig_eval_qi_mitigation.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(art_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ==============================================================================
# FIGURE 4: Multi-Criteria Composite Guardrail Dynamics & Ablation Telemetry
# ==============================================================================
def plot_figure_4(stage2_data, summary_data):
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.9, 1.4])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # 4A: Ablation Comparison (Approach A vs Approach B)
    ablation = summary_data['guardrail_telemetry']['ablation_comparison']
    approaches = ['Single Threshold\n(Cosine $\\geq 0.80$)', '3-Way Composite\n(Proposed Guardrail)']
    acc_rates = [41.0, 22.0]
    fall_rates = [59.0, 78.0]
    
    x = np.arange(len(approaches))
    bar_w = 0.5
    
    ax1.bar(x, acc_rates, bar_w, label='Accepted Candidate Rewrite', color='#10B981', alpha=0.92, edgecolor='#1E293B')
    ax1.bar(x, fall_rates, bar_w, bottom=acc_rates, label='Fallback to Stage 1 Surrogate', color='#EF4444', alpha=0.92, edgecolor='#1E293B')
    
    for i in range(len(approaches)):
        ax1.text(x[i], acc_rates[i] / 2, f"{acc_rates[i]:.0f}%\nAccepted", ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=10)
        ax1.text(x[i], acc_rates[i] + fall_rates[i] / 2, f"{fall_rates[i]:.0f}%\nFallback", ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=10)
        
    ax1.set_title('A. Guardrail Ablation Comparison', fontweight='bold', pad=12, loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(approaches, fontweight='bold')
    ax1.set_ylabel('Percentage of Documents (N=100)', fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis='y')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), framealpha=0.95, edgecolor='#CBD5E1', fontsize=8.5)
    
    # 4B: Rejection Failure Mode Breakdown (Donut Chart)
    rejection_causes = summary_data['guardrail_telemetry']['rejection_cause_distribution']
    labels = ['Multiple Safety\nFailures (89.7%)', 'Tau Composite\nFailure (10.3%)']
    sizes = [rejection_causes['multiple_failures'], rejection_causes['tau_failure']]
    donut_colors = ['#EF4444', '#F59E0B']
    
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=donut_colors, autopct='%1.1f%%',
        startangle=140, pctdistance=0.68,
        wedgeprops=dict(width=0.48, edgecolor='#FFFFFF', linewidth=2.5)
    )
    for at in autotexts:
        at.set_color('#FFFFFF')
        at.set_fontweight('bold')
        at.set_fontsize(10)
    for t in texts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    
    ax2.set_title('B. Fallback Cause Distribution\n(78 Rejected Candidates)', fontweight='bold', pad=12, loc='left')
    
    # 4C: Candidate Score Distributions & Safety Floors
    scores = []
    for k, v in stage2_data.items():
        scores.append({
            'composite': v.get('composite_score', 0),
            'semantic': v.get('metrics_breakdown', {}).get('semantic_similarity', 0),
            'qi_abs': v.get('metrics_breakdown', {}).get('qi_abstraction_score', 0),
            'readability': v.get('metrics_breakdown', {}).get('readability_score', 0),
            'status': 'Accepted (22%)' if v.get('is_accepted', False) else 'Fallback (78%)'
        })
    df_s = pd.DataFrame(scores)
    
    # Boxplot of metrics with threshold annotations
    metric_cols = ['composite', 'semantic', 'qi_abs', 'readability']
    col_names = ['$S_{\\mathrm{composite}}$', '$S_{\\mathrm{semantic}}$', '$S_{\\mathrm{qi}}$', '$S_{\\mathrm{read}}$']
    
    melted = pd.melt(df_s, id_vars=['status'], value_vars=metric_cols, var_name='Metric', value_name='Score')
    melted['Metric_Name'] = melted['Metric'].map(dict(zip(metric_cols, col_names)))
    
    palette = {'Accepted (22%)': '#10B981', 'Fallback (78%)': '#EF4444'}
    sns.boxplot(data=melted, x='Metric_Name', y='Score', hue='status', ax=ax3, palette=palette, width=0.55,
                linewidth=1.2, flierprops=dict(marker='o', markersize=4, alpha=0.6))
    
    # Add horizontal safety threshold indicators
    # Tau = 0.72
    ax3.axhline(0.72, color='#6366F1', linestyle='--', linewidth=1.5, label='Tau Threshold ($\\tau = 0.72$)')
    # Semantic floor = 0.60
    ax3.axhline(0.60, color='#0284C7', linestyle=':', linewidth=1.5, label='Semantic Floor ($\\mathrm{floor} = 0.60$)')
    # QI floor = 0.50
    ax3.axhline(0.50, color='#D97706', linestyle='-.', linewidth=1.5, label='QI Floor ($\\mathrm{floor} = 0.50$)')
    
    ax3.set_title('C. Candidate Score Distribution & Safety Floors', fontweight='bold', pad=12, loc='left')
    ax3.set_xlabel('Guardrail Metric Dimension', fontweight='bold')
    ax3.set_ylabel('Candidate Metric Value', fontweight='bold')
    ax3.set_ylim(-0.05, 1.15)
    ax3.grid(True, axis='y')
    ax3.legend(loc='lower left', framealpha=0.95, edgecolor='#CBD5E1', fontsize=8, ncol=2)
    
    plt.suptitle('Figure 4: Three-Way Multi-Criteria Composite Guardrail Dynamics & Ablation Telemetry',
                 fontsize=14, fontweight='bold', y=0.98, color='#0F172A')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    out_path = os.path.join(OUTPUT_DIR, 'fig_eval_guardrail_telemetry.png')
    art_path = os.path.join(ARTIFACT_DIR, 'fig_eval_guardrail_telemetry.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(art_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Loading benchmark evaluation data...")
    df_results = pd.read_csv('data/evaluation_100_results.csv')
    df_entity = pd.read_csv('data/evaluation_100_entity_breakdown.csv')
    df_qi = pd.read_csv('data/evaluation_100_qi_breakdown.csv')
    
    with open('data/evaluation_100_summary.json', 'r') as f:
        summary_data = json.load(f)
    with open('data/evaluation_100_stage2_results.json', 'r') as f:
        stage2_data = json.load(f)
        
    print("Generating Figure 1: System Benchmark Comparison Matrix...")
    plot_figure_1(df_results)
    
    print("Generating Figure 2: Granular Direct PII Extraction Performance...")
    plot_figure_2(df_entity)
    
    print("Generating Figure 3: Quasi-Identifier Abstraction & Risk Mitigation...")
    plot_figure_3(df_qi)
    
    print("Generating Figure 4: Multi-Criteria Composite Guardrail Telemetry...")
    plot_figure_4(stage2_data, summary_data)
    
    print("All 4 visualizations generated successfully!")


if __name__ == '__main__':
    main()
