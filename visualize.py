import os
import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path=None):
    if db_path is None:
        db_path = os.path.join(BASE_DIR, "federated_audit.db")
    conn = sqlite3.connect(db_path)
    df_client = pd.read_sql_query("SELECT * FROM fl_metrics ORDER BY round, client_id", conn)
    df_global = pd.read_sql_query("SELECT * FROM global_metrics ORDER BY round", conn)
    conn.close()
    return df_client, df_global

# ==============================================================================
# VISUALIZATION 1: Convergence Dynamics & Client Performance Spread
# ==============================================================================
def plot_convergence_dynamics(df_client, df_global):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    rounds = df_global['round']
    
    # 1A. Training Loss
    for cid in df_client['client_id'].unique():
        c_data = df_client[df_client['client_id'] == cid]
        ax1.plot(c_data['round'], c_data['train_loss'], color='gray', alpha=0.35, linewidth=1.2, linestyle='--')
    
    ax1.plot(rounds, df_global['avg_loss'], color='#1f77b4', linewidth=2.8, marker='o', label='Global Mean Loss')
    ax1.fill_between(rounds, 
                     df_client.groupby('round')['train_loss'].min(), 
                     df_client.groupby('round')['train_loss'].max(), 
                     color='#1f77b4', alpha=0.15, label='Client Range (Min-Max)')
    ax1.set_title("A. Training Loss Convergence", fontweight='bold', pad=10)
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_xticks(rounds)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # 1B. Training Accuracy
    for cid in df_client['client_id'].unique():
        c_data = df_client[df_client['client_id'] == cid]
        ax2.plot(c_data['round'], c_data['train_acc'] * 100, color='gray', alpha=0.35, linewidth=1.2, linestyle='--')
        
    ax2.plot(rounds, df_global['avg_acc'] * 100, color='#2ca02c', linewidth=2.8, marker='s', label='Global Mean Accuracy')
    ax2.fill_between(rounds, 
                     df_client.groupby('round')['train_acc'].min() * 100, 
                     df_client.groupby('round')['train_acc'].max() * 100, 
                     color='#2ca02c', alpha=0.15, label='Client Range (Min-Max)')
    ax2.set_title("B. Classification Accuracy Trajectory", fontweight='bold', pad=10)
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(rounds)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    
    plt.suptitle("Representation 1: Federated Learning Convergence & Performance Dynamics", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "1_convergence_dynamics.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Representation 1: {output_path}")

# ==============================================================================
# VISUALIZATION 2: Explainability Drift & Alignment Tracking (Cosine Sim & KL Div)
# ==============================================================================
def plot_explainability_drift(df_client, df_global):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    rounds = df_global['round']
    
    # 2A. Cosine Similarity (Directional Consistency)
    palette = sns.color_palette("tab10", len(df_client['client_id'].unique()))
    for idx, cid in enumerate(df_client['client_id'].unique()):
        c_data = df_client[df_client['client_id'] == cid]
        ax1.plot(c_data['round'], c_data['cosine_similarity'], marker='.', linewidth=1.5, alpha=0.7, 
                 label=f'Client {cid}', color=palette[idx])
    
    ax1.plot(rounds, df_global['mean_cosine_sim'], color='#d62728', linewidth=3.2, linestyle='-', 
             marker='D', label='Mean Cosine Alignment')
    ax1.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='Ideal Alignment (1.0)')
    ax1.set_title("A. SHAP Feature Attribution Alignment (Cosine Similarity - Higher is Better)", fontweight='bold')
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_ylim(0.7, 1.02)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', ncol=4, frameon=True, facecolor='white', framealpha=0.9)
    
    # 2B. KL Divergence (Distributional Entropy Drift)
    for idx, cid in enumerate(df_client['client_id'].unique()):
        c_data = df_client[df_client['client_id'] == cid]
        ax2.plot(c_data['round'], c_data['kl_divergence'], marker='.', linewidth=1.5, alpha=0.7, 
                 label=f'Client {cid}', color=palette[idx])
        
    ax2.plot(rounds, df_global['mean_kl_div'], color='#9467bd', linewidth=3.2, linestyle='-', 
             marker='^', label='Mean KL Divergence')
    ax2.set_title("B. Explanation Distributional Divergence (Kullback-Leibler Drift - Lower is Better)", fontweight='bold')
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("KL Divergence (nats)")
    ax2.set_xticks(rounds)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', ncol=4, frameon=True, facecolor='white', framealpha=0.9)
    
    plt.suptitle("Representation 2: Explainability Consistency & Concept Drift Tracking", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "2_explainability_drift.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Representation 2: {output_path}")

# ==============================================================================
# VISUALIZATION 3: Client-by-Round Cosine Consistency Matrix (Heatmap)
# ==============================================================================
def plot_client_consistency_heatmap(df_client):
    pivot_cos = df_client.pivot(index="client_id", columns="round", values="cosine_similarity")
    client_means = pivot_cos.mean(axis=1)
    
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    
    # Heatmap
    sns.heatmap(pivot_cos, ax=ax_heat, cmap="viridis", annot=True, fmt=".2f", 
                linewidths=1.0, cbar_kws={'label': 'Cosine Similarity Score', 'orientation': 'horizontal', 'pad': 0.2})
    ax_heat.set_title("A. Client-Wise Cosine Consistency Matrix Across FL Rounds", fontweight='bold', pad=12)
    ax_heat.set_xlabel("Federated Round")
    ax_heat.set_ylabel("Client ID")
    
    # Client Summary Bar
    norm_vals = (client_means - pivot_cos.values.min()) / (pivot_cos.values.max() - pivot_cos.values.min())
    colors = plt.cm.viridis(norm_vals)
    y_pos = np.arange(len(client_means)) + 0.5
    ax_bar.barh(y_pos, client_means, color=colors, edgecolor='black', height=0.6)
    ax_bar.set_xlim(0.7, 1.0)
    ax_bar.set_title("B. Overall Mean", fontweight='bold', pad=12)
    ax_bar.set_xlabel("Mean Similarity")
    for i, val in enumerate(client_means):
        ax_bar.text(val + 0.005, y_pos[i], f"{val:.3f}", va='center', fontsize=9, fontweight='bold')
    ax_bar.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle("Representation 3: Multi-Client Explainability Fidelity Matrix & Disparity Profile", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "3_client_consistency_heatmap.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Representation 3: {output_path}")

# ==============================================================================
# VISUALIZATION 4: Multi-Client SHAP Feature Attribution Spectrum
# ==============================================================================
def plot_feature_attribution_shap(df_client, df_global):
    final_round = int(df_global['round'].max())
    round_data = df_client[df_client['round'] == final_round]
    
    shap_matrix = np.array([json.loads(x) for x in round_data['shap_values']])
    num_clients, num_features = shap_matrix.shape
    features = [f"Feat_{i}" for i in range(num_features)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 4A. Grouped Bar Chart of Feature Importance across Clients at Final Round
    x = np.arange(num_features)
    width = 0.8 / num_clients
    palette = sns.color_palette("tab10", num_clients)
    
    for i in range(num_clients):
        ax1.bar(x + i * width, shap_matrix[i], width=width, label=f'Client {i}', color=palette[i], alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Global mean line
    global_mean_shap = np.mean(shap_matrix, axis=0)
    ax1.plot(x + width * (num_clients - 1) / 2, global_mean_shap, color='black', marker='o', linewidth=2, linestyle='--', label='Consensus Mean')
    
    ax1.set_title(f"A. Local Feature Attributions (Round {final_round})", fontweight='bold', pad=10)
    ax1.set_xlabel("Feature Index")
    ax1.set_ylabel("Mean |SHAP Value| (Attribution Magnitude)")
    ax1.set_xticks(x + width * (num_clients - 1) / 2)
    ax1.set_xticklabels(features, rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    # 4B. Feature Attribution Heatmap Across Clients
    df_shap = pd.DataFrame(shap_matrix, index=[f"Client {i}" for i in range(num_clients)], columns=features)
    sns.heatmap(df_shap, ax=ax2, cmap="YlOrRd", annot=True, fmt=".3f", linewidths=1.0, 
                cbar_kws={'label': 'Mean |SHAP Value|'})
    ax2.set_title(f"B. Feature Attribution Heatmap (Round {final_round})", fontweight='bold', pad=10)
    ax2.set_xlabel("Feature Index")
    ax2.set_xticklabels(features, rotation=45)
    
    plt.suptitle(f"Representation 4: SHAP Feature Attribution Spectrum & Client Discrepancy (Round {final_round})", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "4_feature_attribution_shap.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Representation 4: {output_path}")

# ==============================================================================
# ALL-IN-ONE 4-PANEL DASHBOARD FIGURE
# ==============================================================================
def plot_all_in_one_dashboard(df_client, df_global):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)
    
    rounds = df_global['round']
    
    # 1. Performance Convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(rounds, df_global['avg_loss'], color='#1f77b4', marker='o', linewidth=2.5, label='Global Loss')
    l2 = ax1_twin.plot(rounds, df_global['avg_acc'] * 100, color='#2ca02c', marker='s', linewidth=2.5, label='Global Accuracy (%)')
    ax1.set_title("1. Federated Convergence Dynamics", fontweight='bold', fontsize=13)
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Loss", color='#1f77b4')
    ax1_twin.set_ylabel("Accuracy (%)", color='#2ca02c')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Explainability Consistency & Drift
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    l3 = ax2.plot(rounds, df_global['mean_cosine_sim'], color='#d62728', marker='D', linewidth=2.5, label='Mean Cosine Sim')
    l4 = ax2_twin.plot(rounds, df_global['mean_kl_div'], color='#9467bd', marker='^', linewidth=2.5, label='Mean KL Div')
    ax2.set_title("2. Global XAI Alignment & Drift", fontweight='bold', fontsize=13)
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Cosine Similarity", color='#d62728')
    ax2_twin.set_ylabel("KL Divergence (nats)", color='#9467bd')
    lines2 = l3 + l4
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='center right', frameon=True, facecolor='white')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    pivot_cos = df_client.pivot(index="client_id", columns="round", values="cosine_similarity")
    sns.heatmap(pivot_cos, ax=ax3, cmap="viridis", annot=True, fmt=".2f", cbar=True,
                cbar_kws={'label': 'Cosine Sim', 'orientation': 'horizontal', 'pad': 0.15})
    ax3.set_title("3. Client Cosine Consistency Matrix", fontweight='bold', fontsize=13)
    ax3.set_xlabel("Federated Round")
    ax3.set_ylabel("Client ID")
    
    # 4. Feature Importance
    ax4 = fig.add_subplot(gs[1, 1])
    final_round = int(df_global['round'].max())
    round_data = df_client[df_client['round'] == final_round]
    shap_matrix = np.array([json.loads(x) for x in round_data['shap_values']])
    num_clients, num_features = shap_matrix.shape
    features = [f"Feat_{i}" for i in range(num_features)]
    df_shap = pd.DataFrame(shap_matrix, index=[f"Client {i}" for i in range(num_clients)], columns=features)
    sns.heatmap(df_shap, ax=ax4, cmap="YlOrRd", annot=True, fmt=".3f", cbar=True)
    ax4.set_title(f"4. Local Feature Attributions (Round {final_round})", fontweight='bold', fontsize=13)
    ax4.set_xlabel("Feature Index")
    ax4.set_ylabel("Client ID")
    ax4.set_xticklabels(features, rotation=45)
    
    plt.suptitle("Federated Learning XAI Audit - 4-Panel Analytical Representation", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "combined_dashboard_4in1.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Combined 4-in-1 Overview: {output_path}")

def generate_all_visualizations():
    print("[+] Loading audit database...")
    df_client, df_global = load_data()
    print(f"[+] Loaded {len(df_client)} client records and {len(df_global)} global rounds.")
    
    plot_convergence_dynamics(df_client, df_global)
    plot_explainability_drift(df_client, df_global)
    plot_client_consistency_heatmap(df_client)
    plot_feature_attribution_shap(df_client, df_global)
    plot_all_in_one_dashboard(df_client, df_global)
    print("\n[+] All 4 Visualization Representations generated successfully in the 'plots/' directory!")

if __name__ == "__main__":
    generate_all_visualizations()
