"""论文表格自动生成器
从 benchmark_table.csv / final_stats.csv / benchmark_raw_data.npz
生成论文所需的所有表格: 表5-2 (物理层), 表5-3 (网络层), 表5-4 (最终部署), 表5-5 (统计检验)

用法: python tests/generate_paper_tables.py
输出: images/paper_table_5_2.{csv,tex}, images/paper_table_5_3.{csv,tex},
      images/paper_table_5_4.{csv,tex}, images/paper_table_5_5.{csv,tex}
"""
import os, sys, numpy as np, pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    df = pd.read_csv(os.path.join(BASE, "images", "benchmark_table.csv"), skipinitialspace=True)
    raw = np.load(os.path.join(BASE, "images", "benchmark_raw_data.npz"), allow_pickle=True)
    results = raw['results'].item()
    return df, results


def table_5_2(df, results):
    """物理层指标汇总 (论文表5-2)"""
    methods = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    cols = ['Stress_Mean(N)', 'Stress_P95(N)', 'Stress_Peak(N)',
            'Super_Threshold_Rate(%)', 'Error_RMSE(m)', 'Error_Mean(m)']
    labels = ['Anomaly Mean (N)', 'Anomaly P95 (N)', 'Anomaly Peak (N)',
              'Over-Limit Rate (%)', 'Error RMSE (m)', 'Error Mean (m)']
    rows = []
    for i, mid in enumerate(methods):
        row = df[df['MethodID'] == mid].iloc[0]
        raw_peaks = results[mid]['stress_peaks']
        raw_stress = results[mid]['stress']
        raw_error = results[mid]['error']
        rows.append({
            'Method': mid,
            'Stress Mean': f"{row['Stress_Mean(N)']:.1f} +/- {np.std(raw_stress):.0f}",
            'Stress P95': f"{row['Stress_P95(N)']:.1f}",
            'Stress Peak': f"{np.mean(raw_peaks):.0f} +/- {np.std(raw_peaks):.0f}",
            'Over-Limit(%)': f"{row['Super_Threshold_Rate(%)']:.2f}",
            'Error RMSE': f"{row['Error_RMSE(m)']:.4f}",
            'Error Mean': f"{row['Error_Mean(m)']:.4f} +/- {np.std(raw_error):.4f}",
        })
    tbl = pd.DataFrame(rows)
    return tbl


def table_5_3(df, results):
    """网络层指标汇总 (论文表5-3)"""
    methods = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    rows = []
    for mid in methods:
        row = df[df['MethodID'] == mid].iloc[0]
        raw_rtt = results[mid]['rtt']
        raw_jitter = results[mid]['jitter']
        rows.append({
            'Method': mid,
            'RTT Mean (ms)': f"{row['RTT_Mean(ms)']:.2f} +/- {np.std(raw_rtt):.1f}",
            'RTT P95 (ms)': f"{row['RTT_P95(ms)']:.1f}",
            'Jitter (ms)': f"{row['Jitter(ms)']:.2f} +/- {np.std(raw_jitter):.1f}",
            'Collision (%)': f"{row['CollisionRate(%)']:.2f}",
            'Dead-End (%)': f"{row['DeadEndRate(%)']:.2f}",
        })
    return pd.DataFrame(rows)


def table_5_4(df):
    """最终部署验证 (论文表5-4)"""
    fs = pd.read_csv(os.path.join(BASE, "checkpoints", "final_stats.csv"))
    rows = []
    for _, r in fs.iterrows():
        rows.append({
            'Metric': r['Metric'],
            'Mean': f"{r['Mean']:.4f}",
            'Median': f"{r['Median']:.4f}",
            'P95': f"{r['P95']:.4f}",
            'P99': f"{r['P99']:.4f}",
            'Max': f"{r['Max']:.4f}",
        })
    return pd.DataFrame(rows)


def table_5_5(df, results):
    """统计显著性检验 (论文表5-5) — M1 vs M2-M6 on stress_peaks"""
    methods = ['M2', 'M3', 'M4', 'M5', 'M6']
    m1_peaks = results['M1']['stress_peaks']
    rows = []
    for mid in methods:
        other_peaks = results[mid]['stress_peaks']
        t_stat, p_val = stats.ttest_ind(m1_peaks, other_peaks, equal_var=False)
        u1, u2 = np.mean(m1_peaks), np.mean(other_peaks)
        s1, s2 = np.std(m1_peaks, ddof=1), np.std(other_peaks, ddof=1)
        n1, n2 = len(m1_peaks), len(other_peaks)
        s_pool = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = abs(u1 - u2) / s_pool if s_pool else 0
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        rows.append({
            'Contrast': f'M1 vs {mid}',
            r'\bar{x}_1': f'{u1:.1f}',
            r'\bar{x}_2': f'{u2:.1f}',
            't': f'{t_stat:.3f}',
            'p': f'{p_val:.4f}',
            'Signif.': sig,
            "Cohen's d": f'{d:.4f}',
        })
    return pd.DataFrame(rows)


def load_data(scenario='pressure'):
    csv_path = os.path.join(BASE, "images", f"benchmark_table_{scenario}.csv")
    npz_path = os.path.join(BASE, "images", f"benchmark_raw_data_{scenario}.npz")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE, "images", "benchmark_table.csv")
    if not os.path.exists(npz_path):
        npz_path = os.path.join(BASE, "images", "benchmark_raw_data.npz")
    df = pd.read_csv(csv_path, skipinitialspace=True)
    raw = np.load(npz_path, allow_pickle=True)
    results = raw['results'].item()
    return df, results


def main():
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('--scenario', default='pressure', choices=['normal','pressure','both'],
                        help='Which scenario(s) to generate tables for')
    args = parser.parse_args()
    out_dir = os.path.join(BASE, "images")
    os.makedirs(out_dir, exist_ok=True)

    scenarios = ['normal', 'pressure'] if args.scenario == 'both' else [args.scenario]

    for scenario in scenarios:
        df, results = load_data(scenario)
        print(f"\n{'='*60}")
        print(f"  Generating tables for: {scenario}")
        print(f"{'='*60}")

        tables = {
            'paper_table_5_2': table_5_2,
            'paper_table_5_3': table_5_3,
            'paper_table_5_4': table_5_4,
            'paper_table_5_5': table_5_5,
        }

        for name, fn in tables.items():
            if name == 'paper_table_5_4':
                tbl = fn(df)
            elif name == 'paper_table_5_5':
                tbl = fn(df, results)
            else:
                tbl = fn(df, results)

            tag = f'_{scenario}' if len(scenarios) > 1 else ''
            csv_p = os.path.join(out_dir, f"{name}{tag}.csv")
            tex_p = os.path.join(out_dir, f"{name}{tag}.tex")
            tbl.to_csv(csv_p, index=False)
            with open(tex_p, 'w') as f:
                f.write(tbl.to_latex(index=False, escape=False))
            print(f"  {name}{tag}: generated")
        print(tbl.to_string(index=False))

    print("\nAll paper tables generated.")


if __name__ == "__main__":
    main()
