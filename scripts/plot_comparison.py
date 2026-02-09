"""
M6 Visualization: Compare VidReward vs Baselines (Success Rate)

Reads training logs (Monitor.csv or Tensorboard) and plots aggregated learning curves.
Usage:
    python scripts/plot_comparison.py --dirs runs/residual_throw runs/baselines/bc runs/baselines/pure_rl --labels "VidReward" "BC" "Pure RL"
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def load_monitor_data(run_dir):
    """Load all monitor.csv files from a run directory."""
    monitor_files = glob(os.path.join(run_dir, "**", "monitor.csv"), recursive=True)
    if not monitor_files:
        print(f"Warning: No monitor files found in {run_dir}")
        return None
        
    dfs = []
    for f in monitor_files:
        try:
            # Skip first line (header info)
            df = pd.read_csv(f, skiprows=1)
            # Add cumulative timesteps
            df['timesteps'] = df['l'].cumsum()
            df['success'] = df.get('success', 0.0) # Assume 'success' column exists if logged
            # If explicit success key not in monitor, we might need to rely on 'r' (reward)
            # But for Adroit verify if Monitor wrapper logged 'success'
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return None
        
    # Concatenate all seeds
    return pd.concat(dfs, ignore_index=True)

def plot_comparison(args):
    # sns.set_theme(style="darkgrid")
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'ggplot')
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (run_dir, label) in enumerate(zip(args.dirs, args.labels)):
        print(f"Loading {label} from {run_dir}...")
        df = load_monitor_data(run_dir)
        
        if df is None:
            continue
            
        color = colors[i % len(colors)]
            
        # Success Rate Rolling Average
        df = df.sort_values('timesteps')
        
        # Binning for aggregation
        # Create bins of 2000 steps
        bin_size = 2000
        df['bin'] = (df['timesteps'] // bin_size) * bin_size
        
        # Group by bin
        grouped = df.groupby('bin')['success'].agg(['mean', 'std', 'count']).reset_index()
        # Filter bins with few samples
        grouped = grouped[grouped['count'] > 1]
        
        # Smooth
        window = 5
        mean = grouped['mean'].rolling(window=window, min_periods=1).mean()
        std = grouped['std'].rolling(window=window, min_periods=1).mean()
        x = grouped['bin']
        
        plt.plot(x, mean, label=label, color=color, linewidth=2)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        
    if args.baseline:
        plt.axhline(args.baseline, color='red', linestyle='--', label=f'BC Baseline ({args.baseline:.0%})')
        
    plt.title("VidReward vs Baselines (Catch Task)")
    plt.xlabel("Timesteps")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    
    out_path = "comparison_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs='+', required=True, help="List of run directories")
    parser.add_argument("--labels", nargs='+', required=True, help="List of labels")
    parser.add_argument("--baseline", type=float, help="Fixed baseline value (e.g., 0.2)")
    
    args = parser.parse_args()
    plot_comparison(args)
