"""
Create PnL distribution plots for each agent in each environment.

For each agent-environment combination, creates a distribution plot (histogram + KDE)
highlighting the mean PnL, median, and interquartile range (IQR).
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_results import load_all_metrics

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_pnl_data(results_dir="results"):
    """
    Load all PnL arrays from results directory.
    
    Parameters
    ----------
    results_dir : str
        Directory containing results
        
    Returns
    -------
    dict
        Nested dict: {env_name: {agent_name: pnl_array}}
    """
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    for env_dir in results_path.iterdir():
        if not env_dir.is_dir():
            continue
        
        env_name = env_dir.name
        results[env_name] = {}
        
        for agent_dir in env_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            
            agent_name = agent_dir.name
            pnl_file = agent_dir / "pnls.npy"
            
            if pnl_file.exists():
                try:
                    pnls = np.load(pnl_file)
                    results[env_name][agent_name] = pnls
                except Exception as e:
                    print(f"Warning: Could not load {pnl_file}: {e}")
    
    return results


def calculate_statistics(pnls):
    """
    Calculate statistics for PnL distribution.
    
    Parameters
    ----------
    pnls : array-like
        PnL values
        
    Returns
    -------
    dict
        Dictionary with mean, median, q25, q75, and IQR
    """
    pnls = np.array(pnls)
    pnls = pnls[~np.isnan(pnls)]
    
    if len(pnls) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'iqr': np.nan,
            'std': np.nan
        }
    
    return {
        'mean': np.mean(pnls),
        'median': np.median(pnls),
        'q25': np.percentile(pnls, 25),
        'q75': np.percentile(pnls, 75),
        'iqr': np.percentile(pnls, 75) - np.percentile(pnls, 25),
        'std': np.std(pnls)
    }


def create_pnl_plot(pnls, agent_name, env_name, output_path):
    """
    Create a PnL distribution plot with highlighted statistics.
    
    Parameters
    ----------
    pnls : array-like
        PnL values
    agent_name : str
        Agent name
    env_name : str
        Environment name
    output_path : Path
        Path to save the figure
    """
    pnls = np.array(pnls)
    pnls = pnls[~np.isnan(pnls)]
    
    if len(pnls) == 0:
        print(f"  Warning: No valid PnL data for {agent_name} on {env_name}")
        return False
    
    # Calculate statistics
    stats_dict = calculate_statistics(pnls)
    mean_val = stats_dict['mean']
    median_val = stats_dict['median']
    q25 = stats_dict['q25']
    q75 = stats_dict['q75']
    iqr = stats_dict['iqr']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histogram with KDE overlay
    sns.histplot(
        pnls,
        bins=50,
        kde=True,
        ax=ax,
        color='steelblue',
        alpha=0.6,
        edgecolor='black',
        linewidth=0.5,
        stat='density'
    )
    
    # Get KDE for smooth line
    try:
        kde = stats.gaussian_kde(pnls)
        x_range = np.linspace(pnls.min(), pnls.max(), 300)
        kde_curve = kde(x_range)
        ax.plot(x_range, kde_curve, 'b-', linewidth=2, alpha=0.8, label='KDE')
    except:
        pass
    
    # Highlight IQR with shaded region
    ax.axvspan(q25, q75, alpha=0.3, color='yellow', label=f'IQR: [{q25:.2f}, {q75:.2f}]')
    
    # Highlight median with vertical line
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2.5, label=f'Median: {median_val:.2f}')
    
    # Highlight mean with vertical line
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.2f}')
    
    # Add text annotations for statistics
    y_max = ax.get_ylim()[1]
    y_text = y_max * 0.95
    
    # Format statistics text
    stats_text = (
        f'Mean: {mean_val:.2f}\n'
        f'Median: {median_val:.2f}\n'
        f'IQR: [{q25:.2f}, {q75:.2f}]\n'
        f'Std Dev: {stats_dict["std"]:.2f}\n'
        f'N: {len(pnls)}'
    )
    
    # Add text box with statistics
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'),
        family='monospace'
    )
    
    # Labels and title
    ax.set_xlabel('PnL', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'PnL Distribution: {agent_name} on {env_name}', fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True


def create_all_plots(pnl_data_dict, output_dir):
    """
    Create PnL distribution plots for all agent-environment combinations.
    
    Parameters
    ----------
    pnl_data_dict : dict
        Dictionary: {env_name: {agent_name: pnl_array}}
    output_dir : Path
        Output directory for figures
        
    Returns
    -------
    dict
        Dictionary mapping (env_name, agent_name) to figure filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figure_map = {}
    
    print(f"Creating plots for {len(pnl_data_dict)} environments...")
    print()
    
    for env_name, agents in sorted(pnl_data_dict.items()):
        print(f"  Environment: {env_name}")
        for agent_name, pnls in sorted(agents.items()):
            # Sanitize filenames
            safe_env_name = env_name.replace('/', '_').replace('\\', '_')
            safe_agent_name = agent_name.replace('/', '_').replace('\\', '_')
            
            fig_filename = f'pnl_dist_{safe_env_name}_{safe_agent_name}.png'
            fig_path = output_dir / fig_filename
            
            if create_pnl_plot(pnls, agent_name, env_name, fig_path):
                figure_map[(env_name, agent_name)] = fig_filename
                print(f"    ✓ {agent_name}")
            else:
                print(f"    ✗ {agent_name} (no data)")
        print()
    
    return figure_map


def format_env_name(env_name):
    """
    Format environment name for display.
    
    Parameters
    ----------
    env_name : str
        Environment name (e.g., 'GBMJumpRegimeEnv')
        
    Returns
    -------
    str
        Formatted name (e.g., 'GBM Jump Regime')
    """
    # Remove 'Env' suffix
    name = env_name.replace('Env', '')
    
    # Add spaces before capital letters
    formatted = ''
    for i, char in enumerate(name):
        if i > 0 and char.isupper():
            formatted += ' ' + char
        else:
            formatted += char
    
    return formatted


def generate_markdown_report(figure_map, output_file, figures_dir):
    """
    Generate markdown report with all PnL distribution plots.
    
    Parameters
    ----------
    figure_map : dict
        Dictionary mapping (env_name, agent_name) to figure filenames
    output_file : Path
        Path to output markdown file
    figures_dir : Path
        Directory containing figures (relative to output_file)
    """
    with open(output_file, 'w') as f:
        # Header
        f.write("# PnL Distribution Plots\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Agent-Environment Combinations:** {len(figure_map)}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This document presents PnL (Profit and Loss) distribution plots for each agent ")
        f.write("in each environment. Each plot shows:\n\n")
        f.write("- **Histogram + KDE:** Distribution of PnL values across evaluation episodes\n")
        f.write("- **Mean (Red dashed line):** Average PnL across all episodes\n")
        f.write("- **Median (Green dashed line):** Median PnL value\n")
        f.write("- **IQR (Yellow shaded region):** Interquartile range (25th to 75th percentile)\n")
        f.write("- **Statistics box:** Summary statistics including mean, median, IQR, std dev, and sample size\n\n")
        
        f.write("---\n\n")
        
        # Group by environment
        envs = sorted(set(env for env, _ in figure_map.keys()))
        
        for env_name in envs:
            env_label = format_env_name(env_name)
            f.write(f"## {env_label}\n\n")
            
            # Get all agents for this environment
            env_agents = [(env, agent) for env, agent in figure_map.keys() if env == env_name]
            env_agents = sorted(env_agents, key=lambda x: x[1])  # Sort by agent name
            
            for env, agent in env_agents:
                fig_filename = figure_map[(env, agent)]
                relative_path = figures_dir / fig_filename
                
                f.write(f"### {agent}\n\n")
                f.write(f"![{agent} PnL Distribution on {env_label}]({relative_path})\n\n")
            
            f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("These plots reveal:\n\n")
        f.write("- **Distribution shape:** Skewness, modality, and tail behavior of PnL distributions\n")
        f.write("- **Central tendency:** Comparison between mean and median reveals distribution skewness\n")
        f.write("- **Variability:** IQR and standard deviation indicate risk and consistency\n")
        f.write("- **Outliers:** Extreme values visible in histogram tails\n")
        f.write("- **Performance comparison:** Easy visual comparison across agents and environments\n\n")
        
    print(f"✓ Markdown report generated: {output_file}")


def main():
    """Main function to create PnL distribution plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create PnL distribution plots for each agent-environment combination")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing results")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save output files")
    parser.add_argument("--figures-dir", type=str, default="figures",
                       help="Subdirectory for figures (relative to output-dir)")
    parser.add_argument("--output-file", type=str, default="results/PNL_DISTRIBUTIONS.md",
                       help="Path to output markdown file")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Creating PnL Distribution Plots")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Figures directory: {args.figures_dir}")
    print(f"  Output file: {args.output_file}")
    print(f"  DPI: {args.dpi}")
    print()
    
    # Set DPI
    plt.rcParams['savefig.dpi'] = args.dpi
    
    # Load PnL data
    print("Loading PnL data...")
    pnl_data_dict = load_pnl_data(args.results_dir)
    
    if not pnl_data_dict:
        print("No PnL data found. Exiting.")
        return
    
    total_combinations = sum(len(agents) for agents in pnl_data_dict.values())
    print(f"  Loaded {len(pnl_data_dict)} environments")
    print(f"  Total agent-environment combinations: {total_combinations}")
    print()
    
    # Create output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print("Creating PnL distribution plots...")
    print()
    figure_map = create_all_plots(pnl_data_dict, figures_dir)
    
    if not figure_map:
        print("No plots created. Exiting.")
        return
    
    print(f"Created {len(figure_map)} plots")
    print()
    
    # Generate markdown report
    print("Generating markdown report...")
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate relative path for figures in markdown
    figures_relative = Path(args.figures_dir)
    
    generate_markdown_report(figure_map, output_file, figures_relative)
    
    print()
    print("=" * 70)
    print("✓ PnL distribution plots complete!")
    print(f"  Figures saved to: {figures_dir}")
    print(f"  Markdown report: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
