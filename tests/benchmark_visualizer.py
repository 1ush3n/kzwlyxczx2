import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon

def plot_grouped_bar_chart(df, output_dir):
    metrics = ['Stress_Peak(N)', 'CollisionRate(%)', 'Jitter(ms)']
    titles = ['Peak Stress (N)', 'Packet Loss Rate (%)', 'Jitter (ms)']
    
    x = np.arange(len(df['MethodID']))
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        colors = sns.color_palette("husl", len(df))
        bars = ax.bar(x, df[metric], color=colors, alpha=0.8)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['MethodID'], rotation=45, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_grouped_bar_chart.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '1_grouped_bar_chart.pdf'))
    plt.close()

def plot_stress_sequence(results, methods, output_dir):
    # 完整展示 6 种方法的应力轨迹
    all_method_ids = [m['id'] for m in methods]
    labels = {
        'M1': 'Ours (Full)', 'M2': 'Traditional (SP+Fixed)',
        'M3': 'RL-Only (SP+RL)', 'M4': 'Random-Routing + RL',
        'M5': 'GNN-Only (GNN+Fixed)', 'M6': 'No-Curriculum'
    }
    colors = {
        'M1': '#00FF00', 'M2': '#FF4444', 'M3': '#FFA500',
        'M4': '#FF69B4', 'M5': '#00BFFF', 'M6': '#9370DB'
    }
    
    # 6行子图，每行一个方法
    fig, axes = plt.subplots(len(all_method_ids), 1, figsize=(14, 18), sharex=True)
    
    for i, m_id in enumerate(all_method_ids):
        ax = axes[i]
        if m_id in results and results[m_id]['first_seq'] is not None and len(results[m_id]['first_seq']) > 0:
            seq = results[m_id]['first_seq']
            ax.plot(seq, color=colors.get(m_id, '#FFFFFF'), linewidth=2, label=labels.get(m_id, m_id))
            ax.axhline(y=1000, color='white', linestyle='--', alpha=0.5, label='Threshold (1000N)')
            ax.set_ylabel("Stress (N)", fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.2)
            ax.set_title(f"Episode Trace: {labels.get(m_id, m_id)} (peak={max(seq):.1f}N)", loc='left', fontsize=14)
            # 根据各方法数据范围设置适当的y轴上限
            y_max = max(3000, max(seq) * 1.2)
            ax.set_ylim(0, y_max)
        else:
            ax.text(0.5, 0.5, f'{m_id}: No data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Episode Trace: {labels.get(m_id, m_id)}", loc='left', fontsize=14)

    plt.xlabel("Physics Step (20ms)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_stress_sequence.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '2_stress_sequence.pdf'))
    plt.close()
    
    # 额外生成一个仅包含 M1/M2/M5 的紧凑对比版本(论文核心三方法)
    compact_ids = ['M1', 'M2', 'M5']
    compact_labels = {'M1': 'Ours (Full)', 'M2': 'Traditional', 'M5': 'GNN-Only'}
    compact_colors = {'M1': '#00FF00', 'M2': '#FF4444', 'M5': '#00BFFF'}
    
    fig2, axes2 = plt.subplots(len(compact_ids), 1, figsize=(14, 10), sharex=True)
    for i, m_id in enumerate(compact_ids):
        ax = axes2[i]
        if m_id in results and results[m_id]['first_seq'] is not None and len(results[m_id]['first_seq']) > 0:
            seq = results[m_id]['first_seq']
            ax.plot(seq, color=compact_colors[m_id], linewidth=2, label=compact_labels[m_id])
            ax.axhline(y=1000, color='white', linestyle='--', alpha=0.5)
            ax.set_ylabel("Stress (N)", fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.2)
            ax.set_title(f"Episode Trace: {compact_labels[m_id]}", loc='left', fontsize=14)
            ax.set_ylim(0, max(3000, max(seq) * 1.2))
    plt.xlabel("Physics Step (20ms)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_stress_sequence_compact.png'), dpi=300)
    plt.close()

def plot_radar_chart(df, output_dir):
    metrics = ['Jitter(ms)', 'CollisionRate(%)', 'Stress_Peak(N)', 'Error_RMSE(m)', 'Super_Threshold_Rate(%)']
    
    norm_df = df.copy()
    for m in metrics:
        max_val = norm_df[m].max()
        min_val = norm_df[m].min()
        if max_val == min_val:
            norm_df[m] = 0.5
        else:
            # 越小越好，所以用 (max - val) / (max - min)
            norm_df[m] = (max_val - norm_df[m]) / (max_val - min_val)
            
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = sns.color_palette("husl", len(df))
    for i, row in norm_df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['MethodID'], color=colors[i], linewidth=3, alpha=0.9)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
        
    ax.set_xticks(angles[:-1])
    short_labels = ['Jitter', 'Loss%', 'Peak Stress', 'RMSE', 'Over-Limit%']
    ax.set_xticklabels(short_labels, fontsize=13, fontweight='bold')
    
    plt.title("System Robustness Profile (Larger is Better)", fontsize=18, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)
    
    plt.savefig(os.path.join(output_dir, '3_radar_chart.png'), dpi=300)
    plt.close()

def plot_violin(results, methods, output_dir):
    data = []
    labels = []
    for m in methods:
        m_id = m['id']
        peaks = results[m_id]['stress_peaks']
        data.extend(peaks)
        labels.extend([m_id] * len(peaks))
        
    df = pd.DataFrame({'Method': labels, 'Peak Stress (N)': data})
    
    plt.figure(figsize=(12, 7))
    sns.violinplot(x='Method', y='Peak Stress (N)', data=df, palette="husl", inner="box", cut=0)
    plt.title("Stress Peak Distribution Analysis", fontsize=18, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_violin_plot_stress.png'), dpi=300)
    plt.close()

def plot_jitter_violin(results, methods, output_dir):
    """绘制抖动分布小提琴图，对应论文图 5-4"""
    data = []
    labels = []
    for m in methods:
        m_id = m['id']
        jitter_list = results[m_id]['jitter']
        data.extend(jitter_list)
        labels.extend([m_id] * len(jitter_list))
        
    df = pd.DataFrame({'Method': labels, 'Jitter (ms)': data})
    
    plt.figure(figsize=(12, 7))
    sns.violinplot(x='Method', y='Jitter (ms)', data=df, palette="husl", inner="quartile", cut=0)
    plt.title("Network Jitter Distribution Analysis (Paper Fig 5-4)", fontsize=18, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_violin_plot_jitter.png'), dpi=300)
    plt.close()

def plot_pvalue_heatmap(p_values, methods, output_dir):
    method_ids = [m['id'] for m in methods if m['id'] != 'M1']
    m1_p_vals = [p_values.get(m_id, 1.0) for m_id in method_ids]
    
    p_matrix = np.array([m1_p_vals])
    
    plt.figure(figsize=(12, 4))
    # 使用更加鲜明的渐变色
    ax = sns.heatmap(p_matrix, annot=True, cmap="YlOrRd_r", cbar_kws={'label': 'p-value'},
                     xticklabels=method_ids, yticklabels=['M1 vs.'], fmt=".3f", 
                     annot_kws={"size": 14, "weight": "bold"})
    
    for i, p in enumerate(m1_p_vals):
        if p < 0.05:
            text = "SIGNIFICANT"
            ax.text(i + 0.5, 0.8, text, ha='center', va='center', color='black', fontsize=10, fontweight='bold')

    plt.title("Statistical Significance Map (Welch's t-test)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_pvalue_heatmap.png'), dpi=300)
    plt.close()

def main():
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('--scenario', default='pressure', choices=['normal','pressure'])
    args = parser.parse_args()
    plt.style.use('dark_background')
    
    data_path = f"images/benchmark_raw_data_{args.scenario}.npz"
    csv_path = f"images/benchmark_table_{args.scenario}.csv"
    if not os.path.exists(data_path):
        data_path = "images/benchmark_raw_data.npz"
    if not os.path.exists(csv_path):
        csv_path = "images/benchmark_table.csv"
    output_dir = "images/benchmark_plots"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading benchmark data...")
    if not os.path.exists(data_path) or not os.path.exists(csv_path):
        print("Error: Benchmark data not found. Run benchmark_runner.py first.")
        return
        
    raw_data = np.load(data_path, allow_pickle=True)
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    
    methods = raw_data['methods'].tolist()
    results = raw_data['results'].item()
    p_values = raw_data['p_values'].item()
    
    print("Generating Grouped Bar Chart...")
    plot_grouped_bar_chart(df, output_dir)
    
    print("Generating Stress Sequence...")
    plot_stress_sequence(results, methods, output_dir)
    
    print("Generating Radar Chart...")
    plot_radar_chart(df, output_dir)
    
    print("Generating Violin Plot...")
    plot_violin(results, methods, output_dir)
    
    plot_jitter_violin(results, methods, output_dir)
    plot_pvalue_heatmap(p_values, methods, output_dir)
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
