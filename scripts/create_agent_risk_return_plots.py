"""
Create individual risk-return plots for each agent.

For each agent, creates a scatter plot with return std (standard deviation) on x-axis
and mean PnL on y-axis. Points are color-coded by environment type (ABM, GBM, OU)
and labeled with environment names.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_results import load_all_metrics


# Color scheme for environment types
ENV_TYPE_COLORS = {
    'ABM': '#2E86AB',  # Blue
    'GBM': '#A23B72',  # Purple
    'OU': '#F18F01'    # Orange
}


def get_env_type(env_name):
    """
    Determine environment type from name.
    
    Parameters
    ----------
    env_name : str
        Environment name
        
    Returns
    -------
    str
        Environment type (ABM, GBM, or OU)
    """
    if 'ABM' in env_name:
        return 'ABM'
    elif 'GBM' in env_name:
        return 'GBM'
    elif 'OU' in env_name:
        return 'OU'
    else:
        return 'Unknown'


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


def create_agent_plots(metrics_dict, output_dir):
    """
    Create risk-return plots for each agent.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics: {env_name: {agent_name: metrics_dict}}
    output_dir : Path
        Output directory for figures
        
    Returns
    -------
    dict
        Dictionary mapping agent names to figure file paths
    """
    # Convert to DataFrame
    rows = []
    for env_name, agents in metrics_dict.items():
        for agent_name, metrics in agents.items():
            row = {
                'Environment': env_name,
                'Agent': agent_name,
                'EnvType': get_env_type(env_name),
                **metrics
            }
            rows.append(row)
    
    if not rows:
        print("No data to create plots from.")
        return {}
    
    df = pd.DataFrame(rows)
    
    # Check if required columns exist
    if 'std' not in df.columns or 'mean' not in df.columns:
        print("Required columns 'std' and 'mean' not found in data.")
        return {}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent_figures = {}
    agents = sorted(df['Agent'].unique())
    
    print(f"Creating plots for {len(agents)} agents...")
    
    for agent in agents:
        agent_df = df[df['Agent'] == agent].copy()
        
        if len(agent_df) == 0:
            continue
        
        # Filter out NaN values
        agent_df = agent_df.dropna(subset=['std', 'mean'])
        
        if len(agent_df) == 0:
            print(f"  Skipping {agent}: no valid data")
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Group by environment type for coloring
        for env_type in ['ABM', 'GBM', 'OU']:
            type_df = agent_df[agent_df['EnvType'] == env_type]
            
            if len(type_df) == 0:
                continue
            
            # Plot points
            scatter = ax.scatter(
                type_df['std'],
                type_df['mean'],
                c=ENV_TYPE_COLORS[env_type],
                label=env_type,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )
            
            # Add environment name labels
            for _, row in type_df.iterrows():
                env_label = format_env_name(row['Environment'])
                ax.annotate(
                    env_label,
                    (row['std'], row['mean']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
                )
        
        # Labels and title
        ax.set_xlabel('Return Standard Deviation (Risk)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean PnL (Return)', fontsize=12, fontweight='bold')
        ax.set_title(f'Risk-Return Profile: {agent}', fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        # Sanitize agent name for filename
        safe_agent_name = agent.replace('/', '_').replace('\\', '_')
        fig_filename = f'agent_risk_return_{safe_agent_name}.png'
        fig_path = output_dir / fig_filename
        
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        agent_figures[agent] = fig_filename
        print(f"  Created plot for {agent}")
    
    return agent_figures


def generate_markdown_report(agent_figures, output_file, figures_dir):
    """
    Generate markdown report with agent plots.
    
    Parameters
    ----------
    agent_figures : dict
        Dictionary mapping agent names to figure filenames
    output_file : Path
        Path to output markdown file
    figures_dir : Path
        Directory containing figures (relative to output_file)
    """
    with open(output_file, 'w') as f:
        # Header
        f.write("# Agent Risk-Return Profiles\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Agents:** {len(agent_figures)}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This document presents risk-return profiles for each agent across all environments.\n")
        f.write("Each plot shows:\n\n")
        f.write("- **X-axis:** Return Standard Deviation (Risk)\n")
        f.write("- **Y-axis:** Mean PnL (Return)\n")
        f.write("- **Color coding:** Environment type (ABM=Blue, GBM=Purple, OU=Orange)\n")
        f.write("- **Labels:** Environment names for each data point\n\n")
        
        f.write("Points in the upper-left quadrant represent the best risk-return trade-offs ")
        f.write("(high return, low risk), while points in the lower-right represent poor trade-offs ")
        f.write("(low return, high risk).\n\n")
        
        f.write("---\n\n")
        
        # Plot for each agent
        f.write("## Agent Risk-Return Profiles\n\n")
        
        for agent in sorted(agent_figures.keys()):
            fig_filename = agent_figures[agent]
            relative_path = figures_dir / fig_filename
            
            f.write(f"### {agent}\n\n")
            f.write(f"![{agent} Risk-Return Profile]({relative_path})\n\n")
            f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("These plots reveal:\n\n")
        f.write("- **Agent specialization:** Some agents excel in specific environment types\n")
        f.write("- **Risk-return trade-offs:** The relationship between risk and return varies by environment\n")
        f.write("- **Consistency:** Agents with tighter clusters show more consistent performance\n")
        f.write("- **Optimal environments:** Upper-left points indicate best environment-agent matches\n\n")
        
    print(f"✓ Markdown report generated: {output_file}")


def main():
    """Main function to create agent risk-return plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create risk-return plots for each agent")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing results")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save output files")
    parser.add_argument("--figures-dir", type=str, default="figures",
                       help="Subdirectory for figures (relative to output-dir)")
    parser.add_argument("--output-file", type=str, default="results/AGENT_RISK_RETURN_PROFILES.md",
                       help="Path to output markdown file")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Creating Agent Risk-Return Profiles")
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
    
    # Load data
    print("Loading results...")
    metrics_dict = load_all_metrics(args.results_dir)
    
    if not metrics_dict:
        print("No results found. Exiting.")
        return
    
    print(f"  Loaded {len(metrics_dict)} environments")
    total_experiments = sum(len(agents) for agents in metrics_dict.values())
    print(f"  Total experiments: {total_experiments}")
    print()
    
    # Create output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print("Creating agent plots...")
    print()
    agent_figures = create_agent_plots(metrics_dict, figures_dir)
    
    if not agent_figures:
        print("No plots created. Exiting.")
        return
    
    print()
    print(f"Created {len(agent_figures)} plots")
    print()
    
    # Generate markdown report
    print("Generating markdown report...")
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate relative path for figures in markdown
    # If output_file is in results/ and figures_dir is results/figures/, 
    # relative path from markdown to figures is just figures/
    figures_relative = Path(args.figures_dir)
    
    generate_markdown_report(agent_figures, output_file, figures_relative)
    
    print()
    print("=" * 70)
    print("✓ Agent risk-return profiles complete!")
    print(f"  Figures saved to: {figures_dir}")
    print(f"  Markdown report: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
