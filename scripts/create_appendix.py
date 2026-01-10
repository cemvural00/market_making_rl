"""
Create appendix files with matrix-format results for thesis.

Generates matrices where:
- Rows: Environments
- Columns: Agents (grouped by RL, Analytic, Heuristic) + Category averages
- One matrix per metric (mean PnL, Sharpe, std, var_95, es_95, var_99, es_99, avg_inventory)
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_results import load_all_metrics


def get_agent_categories():
    """Define agent categories."""
    return {
        "RL": ["PPOAgent", "DeepPPOAgent", "LSTMPPOAgent", "SACAgent", "TD3Agent", "LSTMSACAgent"],
        "Analytic": ["ASClosedFormAgent", "ASSimpleHeuristicAgent"],
        "Heuristic": [
            "ZeroIntelligenceAgent", "MidPriceFollowAgent", "NoiseTraderNormal",
            "FixedSpreadAgent", "NoiseTraderUniform", "InventorySpreadScalerAgent",
            "InventoryShiftAgent", "MarketOrderOnlyAgent", "LastLookAgent"
        ]
    }


def format_value(value, metric_name, decimal_places=4):
    """Format a metric value for display."""
    try:
        if value is None:
            return "N/A"
        
        # Check for NaN
        try:
            if isinstance(value, float) and pd.isna(value):
                return "N/A"
        except (TypeError, ValueError):
            pass
        
        if isinstance(value, (int, float)):
            return f"{value:.{decimal_places}f}"
        return str(value)
    except Exception:
        return "N/A"


def create_metric_matrix(df, metric_name, agent_categories, environments):
    """
    Create a matrix for a specific metric.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Results dataframe with columns: Environment, Agent, and metrics
    metric_name : str
        Name of metric (e.g., 'mean', 'sharpe')
    agent_categories : dict
        Dictionary mapping category names to agent lists
    environments : list
        List of environment names
    
    Returns
    -------
    str
        Markdown table string
    """
    if metric_name not in df.columns:
        return None
    
    # Get all available agents from dataframe
    available_agents = set(df["Agent"].unique())
    
    # Build header: RL agents, RL avg, Analytic agents, Analytic avg, Heuristic agents, Heuristic avg
    header = ["Environment"]
    
    # Add RL agents (only if they exist in data)
    rl_agents_in_data = [agent for agent in agent_categories["RL"] if agent in available_agents]
    for agent in rl_agents_in_data:
        header.append(agent)
    header.append("**RL Avg**")
    
    # Add Analytic agents (only if they exist in data)
    analytic_agents_in_data = [agent for agent in agent_categories["Analytic"] if agent in available_agents]
    for agent in analytic_agents_in_data:
        header.append(agent)
    header.append("**Analytic Avg**")
    
    # Add Heuristic agents (only if they exist in data)
    heuristic_agents_in_data = [agent for agent in agent_categories["Heuristic"] if agent in available_agents]
    for agent in heuristic_agents_in_data:
        header.append(agent)
    header.append("**Heuristic Avg**")
    
    # Create header row
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    
    # Build rows for each environment
    for env_name in sorted(environments):
        env_df = df[df["Environment"] == env_name].copy()
        row = [env_name]
        
        # RL agents values and average
        rl_values = []
        for agent in rl_agents_in_data:
            agent_data = env_df[env_df["Agent"] == agent]
            if not agent_data.empty and metric_name in agent_data.columns:
                value = agent_data[metric_name].iloc[0]
                rl_values.append(value)
                row.append(format_value(value, metric_name))
            else:
                row.append("N/A")
        
        # Calculate RL average
        valid_rl_values = [v for v in rl_values if isinstance(v, (int, float)) and not pd.isna(v)]
        if valid_rl_values:
            rl_avg = sum(valid_rl_values) / len(valid_rl_values)
            row.append(format_value(rl_avg, metric_name))
        else:
            row.append("N/A")
        
        # Analytic agents values and average
        analytic_values = []
        for agent in analytic_agents_in_data:
            agent_data = env_df[env_df["Agent"] == agent]
            if not agent_data.empty and metric_name in agent_data.columns:
                value = agent_data[metric_name].iloc[0]
                analytic_values.append(value)
                row.append(format_value(value, metric_name))
            else:
                row.append("N/A")
        
        # Calculate Analytic average
        valid_analytic_values = [v for v in analytic_values if isinstance(v, (int, float)) and not pd.isna(v)]
        if valid_analytic_values:
            analytic_avg = sum(valid_analytic_values) / len(valid_analytic_values)
            row.append(format_value(analytic_avg, metric_name))
        else:
            row.append("N/A")
        
        # Heuristic agents values and average
        heuristic_values = []
        for agent in heuristic_agents_in_data:
            agent_data = env_df[env_df["Agent"] == agent]
            if not agent_data.empty and metric_name in agent_data.columns:
                value = agent_data[metric_name].iloc[0]
                heuristic_values.append(value)
                row.append(format_value(value, metric_name))
            else:
                row.append("N/A")
        
        # Calculate Heuristic average
        valid_heuristic_values = [v for v in heuristic_values if isinstance(v, (int, float)) and not pd.isna(v)]
        if valid_heuristic_values:
            heuristic_avg = sum(valid_heuristic_values) / len(valid_heuristic_values)
            row.append(format_value(heuristic_avg, metric_name))
        else:
            row.append("N/A")
        
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def create_appendix(results_dir="results", output_file="results/APPENDIX.md"):
    """
    Create appendix file with matrix-format results.
    
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
    
    # Convert to DataFrame
    rows = []
    for env_name, agents in results.items():
        for agent_name, metrics in agents.items():
            row = {
                "Environment": env_name,
                "Agent": agent_name,
                **metrics
            }
            rows.append(row)
    
    if not rows:
        print("No results found to process.")
        return
    
    df = pd.DataFrame(rows)
    
    # Get agent categories
    agent_categories = get_agent_categories()
    
    # Get environments
    environments = sorted(df["Environment"].unique())
    
    # Get all metrics (exclude Environment and Agent columns)
    metrics = [col for col in df.columns if col not in ["Environment", "Agent"]]
    
    # Metric display names
    metric_names = {
        "mean": "Mean PnL",
        "sharpe": "Sharpe Ratio",
        "std": "Standard Deviation",
        "var_95": "Value at Risk (95%)",
        "es_95": "Expected Shortfall (95%)",
        "var_99": "Value at Risk (99%)",
        "es_99": "Expected Shortfall (99%)",
        "avg_inventory": "Average Inventory"
    }
    
    # Create output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        # Header
        f.write("# Appendix: Detailed Performance Matrices\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(df)}\n\n")
        f.write(f"**Environments:** {len(environments)}\n\n")
        f.write(f"**Agents:** {len(df['Agent'].unique())}\n\n")
        
        # Agent category breakdown
        f.write("## Agent Categories\n\n")
        f.write(f"- **RL Agents:** {len(agent_categories['RL'])} - {', '.join(agent_categories['RL'])}\n")
        f.write(f"- **Analytic Agents:** {len(agent_categories['Analytic'])} - {', '.join(agent_categories['Analytic'])}\n")
        f.write(f"- **Heuristic Agents:** {len(agent_categories['Heuristic'])} - {', '.join(agent_categories['Heuristic'])}\n\n")
        
        # Create matrix for each metric
        for metric in metrics:
            metric_display_name = metric_names.get(metric, metric.replace("_", " ").title())
            f.write(f"## {metric_display_name}\n\n")
            
            matrix = create_metric_matrix(df, metric, agent_categories, environments)
            if matrix:
                f.write(matrix)
                f.write("\n\n")
            else:
                f.write(f"*Metric '{metric}' not available in results.*\n\n")
        
        # Overall summary section
        f.write("---\n\n")
        f.write("## Summary Statistics\n\n")
        
        # Calculate overall category averages
        rl_df = df[df["Agent"].isin(agent_categories["RL"])]
        analytic_df = df[df["Agent"].isin(agent_categories["Analytic"])]
        heuristic_df = df[df["Agent"].isin(agent_categories["Heuristic"])]
        
        f.write("### Category Averages Across All Environments\n\n")
        f.write("| Category | ")
        
        # Header for metrics
        for metric in metrics:
            metric_display_name = metric_names.get(metric, metric.replace("_", " ").title())
            f.write(f"{metric_display_name} | ")
        f.write("\n| --- | ")
        for _ in metrics:
            f.write("--- | ")
        f.write("\n")
        
        # RL average row
        f.write("| **RL** | ")
        for metric in metrics:
            if metric in rl_df.columns:
                avg = rl_df[metric].mean()
                f.write(f"{format_value(avg, metric)} | ")
            else:
                f.write("N/A | ")
        f.write("\n")
        
        # Analytic average row
        f.write("| **Analytic** | ")
        for metric in metrics:
            if metric in analytic_df.columns:
                avg = analytic_df[metric].mean()
                f.write(f"{format_value(avg, metric)} | ")
            else:
                f.write("N/A | ")
        f.write("\n")
        
        # Heuristic average row
        f.write("| **Heuristic** | ")
        for metric in metrics:
            if metric in heuristic_df.columns:
                avg = heuristic_df[metric].mean()
                f.write(f"{format_value(avg, metric)} | ")
            else:
                f.write("N/A | ")
        f.write("\n\n")
        
        f.write("---\n\n")
        f.write("*Appendix generated from evaluation results.*\n")
    
    print(f"✓ Appendix saved to: {output_path}")
    return output_path


def main():
    """Main function to create appendix."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create appendix files with matrix-format results")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing results")
    parser.add_argument("--output-file", type=str, default="results/APPENDIX.md",
                       help="Path to output appendix file")
    
    args = parser.parse_args()
    
    print("Creating appendix files...")
    create_appendix(args.results_dir, args.output_file)
    print("\n✓ Appendix generation complete!")


if __name__ == "__main__":
    main()
