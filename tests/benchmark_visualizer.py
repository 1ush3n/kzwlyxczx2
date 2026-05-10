import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon

def plot_grouped_bar_chart(df, output_dir):
    plt.figure(figsize=(12, 6))
    
    metrics = ['Stress_Peak(N)', 'Error_RMSE(m)', 'RTT_Mean(ms)']
    titles = ['Peak Stress (N)', 'RMSE Error (m)', 'Mean RTT (ms)']
    
    x = np.arange(len(df['MethodID']))
    width = 0.25
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        bars = ax.bar(x, df[metric], color=sns.color_palette("husl", len(df)))
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df['MethodID'], rotation=45)
        
        # Add values on top
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_grouped_bar_chart.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '1_grouped_bar_chart.pdf'))
    plt.close()

def plot_stress_sequence(results, methods, output_dir):
    plt.figure(figsize=(14, 6))
    
    target_methods = ['M1', 'M2', 'M3']
    colors = {'M1': 'lime', 'M2': 'red', 'M3': 'orange'}
    labels = {'M1': 'Ours (Full)', 'M2': 'Traditional (Fixed)', 'M3': 'RL-Only'}
    
    for m_id in target_methods:
        if m_id in results:
            seq = results[m_id]['first_seq']
            plt.plot(seq, label=labels[m_id], color=colors[m_id], alpha=0.8, linewidth=2)
            
    plt.axhline(y=1000, color='red', linestyle='--', label='Warning Threshold (1000N)')
    plt.title("Stress Sequence Comparison (Single Episode)", fontsize=16)
    plt.xlabel("Physics Step (20ms)", fontsize=12)
    plt.ylabel("Absolute Stress (N)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_stress_sequence.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '2_stress_sequence.pdf'))
    plt.close()

def plot_radar_chart(df, output_dir):
    metrics = ['RTT_Mean(ms)', 'Jitter(ms)', 'Stress_Peak(N)', 'Error_RMSE(m)', 'Super_Threshold_Rate(%)']
    
    # Normalize data (lower is better for all these metrics, so we do 1 - normalized to make larger area = better)
    norm_df = df.copy()
    for m in metrics:
        max_val = norm_df[m].max()
        min_val = norm_df[m].min()
        if max_val == min_val:
            norm_df[m] = 1.0
        else:
            norm_df[m] = 1.0 - (norm_df[m] - min_val) / (max_val - min_val)
            
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = sns.color_palette("husl", len(df))
    for i, row in norm_df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['MethodID'], color=colors[i], linewidth=2)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
        
    ax.set_xticks(angles[:-1])
    # Shorten labels
    short_labels = ['RTT', 'Jitter', 'Peak Stress', 'RMSE', 'Over-Threshold']
    ax.set_xticklabels(short_labels, fontsize=12)
    
    plt.title("Normalized Performance (Larger Area = Better)", fontsize=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_radar_chart.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '3_radar_chart.pdf'))
    plt.close()

def plot_violin(results, methods, output_dir):
    plt.figure(figsize=(12, 6))
    
    data = []
    labels = []
    
    for m in methods:
        m_id = m['id']
        peaks = results[m_id]['stress_peaks']
        data.extend(peaks)
        labels.extend([m_id] * len(peaks))
        
    df = pd.DataFrame({'Method': labels, 'Peak Stress (N)': data})
    
    sns.violinplot(x='Method', y='Peak Stress (N)', data=df, palette="husl", inner="quartile")
    plt.title("Distribution of Peak Stress Across Episodes", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_violin_plot.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '4_violin_plot.pdf'))
    plt.close()

def plot_pvalue_heatmap(p_values, methods, output_dir):
    method_ids = [m['id'] for m in methods]
    
    # Create an NxN matrix, but we only have M1 vs others. 
    # For a full heatmap, we'd need pairwise. Here we just show a 1xN heatmap.
    
    m1_p_vals = []
    for m_id in method_ids:
        m1_p_vals.append(p_values.get(m_id, 1.0))
        
    p_matrix = np.array([m1_p_vals])
    
    plt.figure(figsize=(10, 3))
    ax = sns.heatmap(p_matrix, annot=True, cmap="coolwarm_r", cbar_kws={'label': 'p-value'},
                     xticklabels=method_ids, yticklabels=['M1 vs.'])
    
    # Add significance stars
    for i in range(len(method_ids)):
        p = p_matrix[0, i]
        if p < 0.001:
            text = "***"
        elif p < 0.01:
            text = "**"
        elif p < 0.05:
            text = "*"
        else:
            text = "ns"
        ax.text(i + 0.5, 0.2, text, ha='center', va='center', color='white' if p < 0.05 else 'black', fontsize=14, fontweight='bold')

    plt.title("Statistical Significance (Welch's t-test on Peak Stress)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_pvalue_heatmap.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '5_pvalue_heatmap.pdf'))
    plt.close()

def main():
    plt.style.use('dark_background')
    
    data_path = "images/benchmark_raw_data.npz"
    csv_path = "images/benchmark_table.csv"
    output_dir = "images/benchmark_plots"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading benchmark data...")
    if not os.path.exists(data_path) or not os.path.exists(csv_path):
        print("Error: Benchmark data not found. Run benchmark_runner.py first.")
        return
        
    raw_data = np.load(data_path, allow_pickle=True)
    df = pd.read_csv(csv_path)
    
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
    
    print("Generating p-value Heatmap...")
    plot_pvalue_heatmap(p_values, methods, output_dir)
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
