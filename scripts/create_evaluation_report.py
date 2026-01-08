"""
Create a comprehensive evaluation report from existing comparison results.

This script analyzes all available results and creates a detailed markdown report.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_results import load_all_metrics


def create_evaluation_report(results_dir="results", output_file="results/EVALUATION_REPORT.md"):
    """
    Create a comprehensive evaluation report from results.
    
    Parameters
    ----------
    results_dir : str
        Directory containing results
    output_file : str
        Path to output markdown file
    """
    print("Loading results...")
    results = load_all_metrics(results_dir)
    
    if not results:
        print("No results found.")
        return
    
    # Convert to DataFrame for easier analysis
    rows = []
    for env_name, agents in results.items():
        for agent_name, metrics in agents.items():
            row = {
                "Environment": env_name,
                "Agent": agent_name,
                **metrics
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Determine agent categories
    rl_agents = ["PPOAgent", "DeepPPOAgent", "LSTMPPOAgent", "SACAgent", "TD3Agent", "LSTMSACAgent"]
    analytic_agents = ["ASClosedFormAgent", "ASSimpleHeuristicAgent"]
    heuristic_agents = [a for a in df["Agent"].unique() if a not in rl_agents + analytic_agents]
    
    # Create report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        # Header
        f.write("# Market Making Agent Evaluation Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(df)}\n\n")
        f.write(f"**Environments:** {len(df['Environment'].unique())}\n\n")
        f.write(f"**Agents:** {len(df['Agent'].unique())}\n\n")
        
        # Agent breakdown
        f.write("## Agent Categories\n\n")
        f.write(f"- **RL Agents:** {len(rl_agents)} ({', '.join(rl_agents)})\n")
        f.write(f"- **Analytic Agents:** {len(analytic_agents)} ({', '.join(analytic_agents)})\n")
        f.write(f"- **Heuristic Agents:** {len(heuristic_agents)} ({', '.join(heuristic_agents)})\n\n")
        
        # Environment breakdown
        f.write("## Environment Types\n\n")
        env_types = defaultdict(list)
        for env in df['Environment'].unique():
            if 'ABM' in env:
                env_types['ABM'].append(env)
            elif 'GBM' in env:
                env_types['GBM'].append(env)
            elif 'OU' in env:
                env_types['OU'].append(env)
        
        for env_type, envs in env_types.items():
            f.write(f"- **{env_type}:** {len(envs)} environments\n")
        f.write("\n")
        
        # Overall statistics
        f.write("## Overall Performance Statistics\n\n")
        if "mean" in df.columns:
            f.write(f"- **Best Mean PnL:** {df['mean'].max():.4f} ({df.loc[df['mean'].idxmax(), 'Agent']} on {df.loc[df['mean'].idxmax(), 'Environment']})\n")
            f.write(f"- **Worst Mean PnL:** {df['mean'].min():.4f} ({df.loc[df['mean'].idxmin(), 'Agent']} on {df.loc[df['mean'].idxmin(), 'Environment']})\n")
            f.write(f"- **Average Mean PnL:** {df['mean'].mean():.4f}\n")
            f.write(f"- **Std Dev of Mean PnL:** {df['mean'].std():.4f}\n\n")
        
        if "sharpe" in df.columns:
            f.write(f"- **Best Sharpe Ratio:** {df['sharpe'].max():.4f} ({df.loc[df['sharpe'].idxmax(), 'Agent']} on {df.loc[df['sharpe'].idxmax(), 'Environment']})\n")
            f.write(f"- **Worst Sharpe Ratio:** {df['sharpe'].min():.4f} ({df.loc[df['sharpe'].idxmin(), 'Agent']} on {df.loc[df['sharpe'].idxmin(), 'Environment']})\n")
            f.write(f"- **Average Sharpe Ratio:** {df['sharpe'].mean():.4f}\n\n")
        
        # Best agent per environment
        f.write("## Best Agent per Environment\n\n")
        f.write("### By Mean PnL\n\n")
        f.write("| Environment | Best Agent | Mean PnL | Sharpe | Std Dev |\n")
        f.write("|-------------|------------|----------|--------|----------|\n")
        for env_name in sorted(df['Environment'].unique()):
            env_df = df[df['Environment'] == env_name].copy()
            if "mean" in env_df.columns:
                best = env_df.loc[env_df['mean'].idxmax()]
                sharpe = best.get('sharpe', 'N/A')
                std = best.get('std', 'N/A')
                if isinstance(sharpe, (int, float)):
                    sharpe = f"{sharpe:.4f}"
                if isinstance(std, (int, float)):
                    std = f"{std:.4f}"
                f.write(f"| {env_name} | {best['Agent']} | {best['mean']:.4f} | {sharpe} | {std} |\n")
        f.write("\n")
        
        f.write("### By Sharpe Ratio\n\n")
        f.write("| Environment | Best Agent | Sharpe | Mean PnL | Std Dev |\n")
        f.write("|-------------|------------|--------|----------|----------|\n")
        for env_name in sorted(df['Environment'].unique()):
            env_df = df[df['Environment'] == env_name].copy()
            if "sharpe" in env_df.columns:
                best = env_df.loc[env_df['sharpe'].idxmax()]
                mean = best.get('mean', 'N/A')
                std = best.get('std', 'N/A')
                if isinstance(mean, (int, float)):
                    mean = f"{mean:.4f}"
                if isinstance(std, (int, float)):
                    std = f"{std:.4f}"
                f.write(f"| {env_name} | {best['Agent']} | {best['sharpe']:.4f} | {mean} | {std} |\n")
        f.write("\n")
        
        # Best environment per agent
        f.write("## Best Environment per Agent\n\n")
        f.write("### By Mean PnL\n\n")
        f.write("| Agent | Best Environment | Mean PnL | Sharpe | Std Dev |\n")
        f.write("|-------|------------------|----------|--------|----------|\n")
        for agent_name in sorted(df['Agent'].unique()):
            agent_df = df[df['Agent'] == agent_name].copy()
            if "mean" in agent_df.columns:
                best = agent_df.loc[agent_df['mean'].idxmax()]
                sharpe = best.get('sharpe', 'N/A')
                std = best.get('std', 'N/A')
                if isinstance(sharpe, (int, float)):
                    sharpe = f"{sharpe:.4f}"
                if isinstance(std, (int, float)):
                    std = f"{std:.4f}"
                f.write(f"| {agent_name} | {best['Environment']} | {best['mean']:.4f} | {sharpe} | {std} |\n")
        f.write("\n")
        
        # Agent category performance
        f.write("## Performance by Agent Category\n\n")
        
        # RL agents
        f.write("### RL Agents\n\n")
        rl_df = df[df['Agent'].isin(rl_agents)].copy()
        if not rl_df.empty and "mean" in rl_df.columns:
            f.write(f"- **Average Mean PnL:** {rl_df['mean'].mean():.4f}\n")
            if 'sharpe' in rl_df.columns:
                f.write(f"- **Average Sharpe:** {rl_df['sharpe'].mean():.4f}\n")
            else:
                f.write(f"- **Average Sharpe:** N/A\n")
            f.write(f"- **Best RL Agent:** {rl_df.loc[rl_df['mean'].idxmax(), 'Agent']} (mean: {rl_df['mean'].max():.4f})\n\n")
        
        # Analytic agents
        f.write("### Analytic Agents\n\n")
        analytic_df = df[df['Agent'].isin(analytic_agents)].copy()
        if not analytic_df.empty and "mean" in analytic_df.columns:
            f.write(f"- **Average Mean PnL:** {analytic_df['mean'].mean():.4f}\n")
            if 'sharpe' in analytic_df.columns:
                f.write(f"- **Average Sharpe:** {analytic_df['sharpe'].mean():.4f}\n")
            else:
                f.write(f"- **Average Sharpe:** N/A\n")
            f.write(f"- **Best Analytic Agent:** {analytic_df.loc[analytic_df['mean'].idxmax(), 'Agent']} (mean: {analytic_df['mean'].max():.4f})\n\n")
        
        # Heuristic agents
        f.write("### Heuristic Agents\n\n")
        heuristic_df = df[df['Agent'].isin(heuristic_agents)].copy()
        if not heuristic_df.empty and "mean" in heuristic_df.columns:
            f.write(f"- **Average Mean PnL:** {heuristic_df['mean'].mean():.4f}\n")
            if 'sharpe' in heuristic_df.columns:
                f.write(f"- **Average Sharpe:** {heuristic_df['sharpe'].mean():.4f}\n")
            else:
                f.write(f"- **Average Sharpe:** N/A\n")
            f.write(f"- **Best Heuristic Agent:** {heuristic_df.loc[heuristic_df['mean'].idxmax(), 'Agent']} (mean: {heuristic_df['mean'].max():.4f})\n\n")
        
        # Environment type performance
        f.write("## Performance by Environment Type\n\n")
        
        for env_type, envs in env_types.items():
            f.write(f"### {env_type} Environments\n\n")
            type_df = df[df['Environment'].isin(envs)].copy()
            if not type_df.empty and "mean" in type_df.columns:
                f.write(f"- **Average Mean PnL:** {type_df['mean'].mean():.4f}\n")
                if 'sharpe' in type_df.columns:
                    f.write(f"- **Average Sharpe:** {type_df['sharpe'].mean():.4f}\n")
                else:
                    f.write(f"- **Average Sharpe:** N/A\n")
                f.write(f"- **Best Agent:** {type_df.loc[type_df['mean'].idxmax(), 'Agent']} on {type_df.loc[type_df['mean'].idxmax(), 'Environment']} (mean: {type_df['mean'].max():.4f})\n\n")
        
        # Risk metrics analysis
        f.write("## Risk Analysis\n\n")
        if "var_95" in df.columns and "es_95" in df.columns:
            f.write("### Value at Risk (VaR) and Expected Shortfall (ES)\n\n")
            f.write("| Agent | Avg VaR (95%) | Avg ES (95%) | Avg VaR (99%) | Avg ES (99%) |\n")
            f.write("|-------|---------------|--------------|---------------|--------------|\n")
            for agent_name in sorted(df['Agent'].unique()):
                agent_df = df[df['Agent'] == agent_name].copy()
                avg_var_95 = agent_df['var_95'].mean() if 'var_95' in agent_df.columns else 'N/A'
                avg_es_95 = agent_df['es_95'].mean() if 'es_95' in agent_df.columns else 'N/A'
                avg_var_99 = agent_df['var_99'].mean() if 'var_99' in agent_df.columns else 'N/A'
                avg_es_99 = agent_df['es_99'].mean() if 'es_99' in agent_df.columns else 'N/A'
                
                if isinstance(avg_var_95, (int, float)):
                    avg_var_95 = f"{avg_var_95:.4f}"
                if isinstance(avg_es_95, (int, float)):
                    avg_es_95 = f"{avg_es_95:.4f}"
                if isinstance(avg_var_99, (int, float)):
                    avg_var_99 = f"{avg_var_99:.4f}"
                if isinstance(avg_es_99, (int, float)):
                    avg_es_99 = f"{avg_es_99:.4f}"
                
                f.write(f"| {agent_name} | {avg_var_95} | {avg_es_95} | {avg_var_99} | {avg_es_99} |\n")
            f.write("\n")
        
        # Inventory management
        f.write("## Inventory Management\n\n")
        if "avg_inventory" in df.columns:
            f.write("### Average Inventory Levels\n\n")
            f.write("| Agent | Avg Inventory (across all envs) |\n")
            f.write("|-------|----------------------------------|\n")
            for agent_name in sorted(df['Agent'].unique()):
                agent_df = df[df['Agent'] == agent_name].copy()
                avg_inv = agent_df['avg_inventory'].mean()
                f.write(f"| {agent_name} | {avg_inv:.4f} |\n")
            f.write("\n")
        
        # Detailed per-environment tables
        f.write("## Detailed Per-Environment Results\n\n")
        for env_name in sorted(df['Environment'].unique()):
            f.write(f"### {env_name}\n\n")
            env_df = df[df['Environment'] == env_name].copy()
            env_df = env_df.drop(columns=["Environment"]).set_index("Agent")
            
            # Sort by mean PnL descending
            if "mean" in env_df.columns:
                env_df = env_df.sort_values("mean", ascending=False)
            
            # Create formatted table
            f.write("| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |\n")
            f.write("|-------|------|-----|--------|-----------|----------|---------------|\n")
            for agent_name, row in env_df.iterrows():
                mean = row.get('mean', 'N/A')
                std = row.get('std', 'N/A')
                sharpe = row.get('sharpe', 'N/A')
                var_95 = row.get('var_95', 'N/A')
                es_95 = row.get('es_95', 'N/A')
                avg_inv = row.get('avg_inventory', 'N/A')
                
                if isinstance(mean, (int, float)):
                    mean = f"{mean:.4f}"
                if isinstance(std, (int, float)):
                    std = f"{std:.4f}"
                if isinstance(sharpe, (int, float)):
                    sharpe = f"{sharpe:.4f}"
                if isinstance(var_95, (int, float)):
                    var_95 = f"{var_95:.4f}"
                if isinstance(es_95, (int, float)):
                    es_95 = f"{es_95:.4f}"
                if isinstance(avg_inv, (int, float)):
                    avg_inv = f"{avg_inv:.4f}"
                
                f.write(f"| {agent_name} | {mean} | {std} | {sharpe} | {var_95} | {es_95} | {avg_inv} |\n")
            f.write("\n")
        
        # Key insights
        f.write("## Key Insights and Observations\n\n")
        
        # Best overall agent
        if "mean" in df.columns:
            best_overall = df.loc[df['mean'].idxmax()]
            f.write(f"1. **Best Overall Performance:** {best_overall['Agent']} achieves the highest mean PnL ({best_overall['mean']:.4f}) on {best_overall['Environment']}.\n\n")
        
        # Most consistent agent
        if "std" in df.columns:
            agent_stds = df.groupby('Agent')['std'].mean()
            most_consistent = agent_stds.idxmin()
            f.write(f"2. **Most Consistent Agent:** {most_consistent} has the lowest average standard deviation ({agent_stds[most_consistent]:.4f}), indicating more stable performance.\n\n")
        
        # Best risk-adjusted return
        if "sharpe" in df.columns:
            best_sharpe = df.loc[df['sharpe'].idxmax()]
            f.write(f"3. **Best Risk-Adjusted Return:** {best_sharpe['Agent']} achieves the highest Sharpe ratio ({best_sharpe['sharpe']:.4f}) on {best_sharpe['Environment']}.\n\n")
        
        # RL vs Analytic vs Heuristic comparison
        if "mean" in df.columns:
            rl_mean = rl_df['mean'].mean() if not rl_df.empty else None
            analytic_mean = analytic_df['mean'].mean() if not analytic_df.empty else None
            heuristic_mean = heuristic_df['mean'].mean() if not heuristic_df.empty else None
            
            f.write("4. **Agent Category Comparison:**\n")
            if rl_mean is not None:
                f.write(f"   - RL Agents average mean PnL: {rl_mean:.4f}\n")
            if analytic_mean is not None:
                f.write(f"   - Analytic Agents average mean PnL: {analytic_mean:.4f}\n")
            if heuristic_mean is not None:
                f.write(f"   - Heuristic Agents average mean PnL: {heuristic_mean:.4f}\n")
            f.write("\n")
        
        # Environment difficulty
        if "mean" in df.columns:
            env_means = df.groupby('Environment')['mean'].mean()
            hardest_env = env_means.idxmin()
            easiest_env = env_means.idxmax()
            f.write(f"5. **Environment Difficulty:**\n")
            f.write(f"   - Most challenging environment: {hardest_env} (avg mean PnL: {env_means[hardest_env]:.4f})\n")
            f.write(f"   - Easiest environment: {easiest_env} (avg mean PnL: {env_means[easiest_env]:.4f})\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated from existing comparison results.*\n")
    
    print(f"✓ Evaluation report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create evaluation report from results")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing results")
    parser.add_argument("--output", type=str, default="results/EVALUATION_REPORT.md",
                       help="Output markdown file path")
    
    args = parser.parse_args()
    
    create_evaluation_report(args.results_dir, args.output)
