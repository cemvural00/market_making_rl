"""
Aggregate and compare results from all agent-environment combinations.

Creates comparison tables and summary reports.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_all_metrics(results_dir="results"):
    """
    Load all metrics from results directory.
    
    Returns
    -------
    dict
        Nested dict: {env_name: {agent_name: metrics_dict}}
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
            metrics_file = agent_dir / "metrics.json"
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                    results[env_name][agent_name] = metrics
                except Exception as e:
                    print(f"Warning: Could not load {metrics_file}: {e}")
    
    return results


def create_comparison_table(results, output_dir="results"):
    """
    Create comparison tables from results.
    
    Parameters
    ----------
    results : dict
        Results from load_all_metrics()
    output_dir : str
        Directory to save output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all data
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
        print("No results found to aggregate.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = output_path / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison table to: {csv_path}")
    
    # Create per-environment comparison
    env_comparison = {}
    for env_name in df["Environment"].unique():
        env_df = df[df["Environment"] == env_name].copy()
        env_df = env_df.drop(columns=["Environment"]).set_index("Agent")
        env_comparison[env_name] = env_df.to_dict(orient="index")
    
    # Create per-agent comparison
    agent_comparison = {}
    for agent_name in df["Agent"].unique():
        agent_df = df[df["Agent"] == agent_name].copy()
        agent_df = agent_df.drop(columns=["Agent"]).set_index("Environment")
        agent_comparison[agent_name] = agent_df.to_dict(orient="index")
    
    # Summary statistics
    summary = {
        "total_experiments": len(rows),
        "environments": sorted(df["Environment"].unique().tolist()),
        "agents": sorted(df["Agent"].unique().tolist()),
        "metrics": [col for col in df.columns if col not in ["Environment", "Agent"]],
        "per_environment": env_comparison,
        "per_agent": agent_comparison
    }
    
    # Save JSON summary
    json_path = output_path / "comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {json_path}")
    
    # Create markdown report
    md_path = output_path / "comparison_report.md"
    create_markdown_report(df, summary, md_path)
    print(f"✓ Saved markdown report to: {md_path}")
    
    return df, summary


def create_markdown_report(df, summary, output_path):
    """Create a markdown report from comparison data."""
    with open(output_path, "w") as f:
        f.write("# Agent Comparison Report\n\n")
        f.write(f"**Total Experiments:** {summary['total_experiments']}\n\n")
        f.write(f"**Environments:** {len(summary['environments'])}\n\n")
        f.write(f"**Agents:** {len(summary['agents'])}\n\n")
        
        f.write("## Environments\n\n")
        for env in summary['environments']:
            f.write(f"- {env}\n")
        
        f.write("\n## Agents\n\n")
        for agent in summary['agents']:
            f.write(f"- {agent}\n")
        
        f.write("\n## Per-Environment Comparison\n\n")
        for env_name in summary['environments']:
            f.write(f"### {env_name}\n\n")
            env_df = df[df["Environment"] == env_name].copy()
            env_df = env_df.drop(columns=["Environment"]).set_index("Agent")
            f.write(env_df.to_markdown())
            f.write("\n\n")
        
        f.write("## Per-Agent Comparison\n\n")
        for agent_name in summary['agents']:
            f.write(f"### {agent_name}\n\n")
            agent_df = df[df["Agent"] == agent_name].copy()
            agent_df = agent_df.drop(columns=["Agent"]).set_index("Environment")
            f.write(agent_df.to_markdown())
            f.write("\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write("### Best Agent per Environment (by Mean PnL)\n\n")
        for env_name in summary['environments']:
            env_df = df[df["Environment"] == env_name].copy()
            if "mean" in env_df.columns:
                best = env_df.loc[env_df["mean"].idxmax()]
                f.write(f"- **{env_name}**: {best['Agent']} (mean PnL: {best['mean']:.4f})\n")
        
        f.write("\n### Best Environment per Agent (by Mean PnL)\n\n")
        for agent_name in summary['agents']:
            agent_df = df[df["Agent"] == agent_name].copy()
            if "mean" in agent_df.columns:
                best = agent_df.loc[agent_df["mean"].idxmax()]
                f.write(f"- **{agent_name}**: {best['Environment']} (mean PnL: {best['mean']:.4f})\n")


def main(results_dir=None, output_dir=None):
    """Main function to aggregate results.

    Parameters
    ----------
    results_dir : str, optional
        Directory containing results. When None, read from CLI args.
    output_dir : str, optional
        Directory to save aggregated results. When None, read from CLI args.
    """
    import argparse

    if results_dir is None or output_dir is None:
        parser = argparse.ArgumentParser(description="Aggregate experiment results")
        parser.add_argument("--results-dir", type=str, default="results",
                           help="Directory containing results")
        parser.add_argument("--output-dir", type=str, default="results",
                           help="Directory to save aggregated results")
        args = parser.parse_args()
        if results_dir is None:
            results_dir = args.results_dir
        if output_dir is None:
            output_dir = args.output_dir

    print("Loading results...")
    results = load_all_metrics(results_dir)

    if not results:
        print("No results found.")
        return

    print(f"Found results for {len(results)} environments")
    total_experiments = sum(len(agents) for agents in results.values())
    print(f"Total experiments: {total_experiments}")

    print("\nCreating comparison tables...")
    df, summary = create_comparison_table(results, output_dir)

    print("\n✓ Aggregation complete!")


if __name__ == "__main__":
    main()
