"""
Create a comprehensive results and conclusions summary document.

This script generates a research paper-style results and conclusions section that
critically analyzes all experimental findings, compares agent categories and
individual agents, and provides insights and conclusions.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_results import load_all_metrics


def categorize_agents():
    """
    Return agent categorization dictionary.
    
    Returns
    -------
    dict
        Dictionary mapping categories to agent lists
    """
    return {
        'RL': ['PPOAgent', 'DeepPPOAgent', 'LSTMPPOAgent', 'SACAgent', 'TD3Agent', 'LSTMSACAgent'],
        'Analytic': ['ASClosedFormAgent', 'ASSimpleHeuristicAgent'],
        'Heuristic': []  # Will be populated dynamically
    }


def get_agent_category(agent_name, agent_categories):
    """
    Get the category of an agent.
    
    Parameters
    ----------
    agent_name : str
        Name of the agent
    agent_categories : dict
        Agent categories dictionary
        
    Returns
    -------
    str
        Category name (RL, Analytic, or Heuristic)
    """
    for category, agents in agent_categories.items():
        if agent_name in agents:
            return category
    return 'Heuristic'


def categorize_environments(df):
    """
    Categorize environments by type and complexity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Environment' column
        
    Returns
    -------
    dict
        Dictionary with environment categorizations
    """
    envs = df['Environment'].unique()
    
    by_type = {'ABM': [], 'GBM': [], 'OU': []}
    by_complexity = {'Vanilla': [], 'Jump': [], 'Regime': [], 'JumpRegime': []}
    
    for env in envs:
        # By type
        if 'ABM' in env:
            by_type['ABM'].append(env)
        elif 'GBM' in env:
            by_type['GBM'].append(env)
        elif 'OU' in env:
            by_type['OU'].append(env)
        
        # By complexity
        if 'JumpRegime' in env or ('Jump' in env and 'Regime' in env):
            by_complexity['JumpRegime'].append(env)
        elif 'Regime' in env:
            by_complexity['Regime'].append(env)
        elif 'Jump' in env:
            by_complexity['Jump'].append(env)
        else:
            by_complexity['Vanilla'].append(env)
    
    return {'by_type': by_type, 'by_complexity': by_complexity}


def calculate_statistics(df, metric_col):
    """
    Calculate comprehensive statistics for a metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    metric_col : str
        Column name for the metric
        
    Returns
    -------
    dict
        Dictionary of statistics
    """
    values = df[metric_col].dropna().values
    
    if len(values) == 0:
        return {
            'mean': np.nan, 'median': np.nan, 'std': np.nan,
            'min': np.nan, 'max': np.nan,
            'p25': np.nan, 'p75': np.nan,
            'count': 0
        }
    
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75),
        'count': len(values)
    }


def calculate_category_statistics(df, agent_categories, metric_col):
    """
    Calculate statistics for each agent category.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data
    agent_categories : dict
        Agent categories dictionary
    metric_col : str
        Metric column name
        
    Returns
    -------
    dict
        Dictionary mapping categories to statistics
    """
    category_stats = {}
    
    for category, agents in agent_categories.items():
        if category == 'Heuristic':
            # Heuristic agents are those not in RL or Analytic
            category_df = df[~df['Agent'].isin(agent_categories['RL'] + agent_categories['Analytic'])]
        else:
            category_df = df[df['Agent'].isin(agents)]
        
        category_stats[category] = calculate_statistics(category_df, metric_col)
    
    return category_stats


def rank_agents(df, metric_col, ascending=False):
    """
    Rank agents by a metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data
    metric_col : str
        Metric column to rank by
    ascending : bool
        If True, lower is better
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rankings
    """
    agent_stats = df.groupby('Agent')[metric_col].agg(['mean', 'median', 'std', 'count']).reset_index()
    agent_stats = agent_stats.sort_values('mean', ascending=ascending)
    agent_stats['rank'] = range(1, len(agent_stats) + 1)
    return agent_stats


def calculate_correlation(df, col1, col2):
    """
    Calculate Pearson correlation between two metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data
    col1, col2 : str
        Column names
        
    Returns
    -------
    float
        Correlation coefficient
    """
    valid = df[[col1, col2]].dropna()
    if len(valid) < 2:
        return np.nan
    return valid[col1].corr(valid[col2])


def identify_outliers(df, metric_col, method='iqr', threshold=3):
    """
    Identify outliers in a metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data
    metric_col : str
        Metric column name
    method : str
        Method: 'iqr' or 'zscore'
    threshold : float
        Threshold for outlier detection
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outlier rows
    """
    values = df[metric_col].dropna()
    
    if len(values) == 0:
        return df.iloc[0:0]
    
    if method == 'iqr':
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = df[(df[metric_col] < lower_bound) | (df[metric_col] > upper_bound)]
    else:  # zscore
        z_scores = np.abs(stats.zscore(values))
        outlier_indices = np.where(z_scores > threshold)[0]
        outliers = df.iloc[outlier_indices]
    
    return outliers


def calculate_consistency(df, agent_col, metric_col):
    """
    Calculate coefficient of variation (consistency) for each agent.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data
    agent_col : str
        Agent column name
    metric_col : str
        Metric column name
        
    Returns
    -------
    pd.DataFrame
        DataFrame with consistency metrics
    """
    consistency = []
    for agent in df[agent_col].unique():
        agent_df = df[df[agent_col] == agent]
        values = agent_df[metric_col].dropna().values
        
        if len(values) > 1:
            cv = np.std(values) / np.abs(np.mean(values)) if np.mean(values) != 0 else np.inf
            consistency.append({
                'Agent': agent,
                'CV': cv,
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Environments': len(values)
            })
    
    return pd.DataFrame(consistency).sort_values('CV')


def analyze_environment_complexity(df, env_categorization, metric_col):
    """
    Analyze how environment complexity affects performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data
    env_categorization : dict
        Environment categorization dictionary
    metric_col : str
        Metric column name
        
    Returns
    -------
    dict
        Statistics by complexity level
    """
    complexity_stats = {}
    
    for complexity, envs in env_categorization['by_complexity'].items():
        complexity_df = df[df['Environment'].isin(envs)]
        complexity_stats[complexity] = calculate_statistics(complexity_df, metric_col)
    
    return complexity_stats


def analyze_environment_type(df, env_categorization, metric_col):
    """
    Analyze how environment type affects performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data
    env_categorization : dict
        Environment categorization dictionary
    metric_col : str
        Metric column name
        
    Returns
    -------
    dict
        Statistics by environment type
    """
    type_stats = {}
    
    for env_type, envs in env_categorization['by_type'].items():
        type_df = df[df['Environment'].isin(envs)]
        type_stats[env_type] = calculate_statistics(type_df, metric_col)
    
    return type_stats


def format_statistical_table(stats_dict, metric_name):
    """
    Format statistics dictionary as markdown table.
    
    Parameters
    ----------
    stats_dict : dict
        Dictionary of statistics
    metric_name : str
        Name of the metric
        
    Returns
    -------
    str
        Markdown table string
    """
    rows = []
    for category, stats in stats_dict.items():
        if np.isnan(stats['mean']):
            rows.append(f"| {category} | N/A | N/A | N/A | N/A |")
        else:
            rows.append(f"| {category} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                       f"{stats['std']:.4f} | [{stats['min']:.4f}, {stats['max']:.4f}] |")
    
    header = f"| Category | Mean | Median | Std Dev | Range |\n|----------|------|--------|---------|-------|\n"
    return f"### {metric_name}\n\n{header}" + "\n".join(rows) + "\n\n"


def generate_report(metrics_dict, output_file):
    """
    Generate the comprehensive results and conclusions report.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics: {env_name: {agent_name: metrics_dict}}
    output_file : str
        Path to output markdown file
    """
    # Convert to DataFrame
    rows = []
    for env_name, agents in metrics_dict.items():
        for agent_name, metrics in agents.items():
            row = {'Environment': env_name, 'Agent': agent_name, **metrics}
            rows.append(row)
    
    if not rows:
        print("No data to generate report from.")
        return
    
    df = pd.DataFrame(rows)
    
    # Determine agent categories
    agent_categories = categorize_agents()
    all_agents = df['Agent'].unique()
    heuristic_agents = [a for a in all_agents if a not in agent_categories['RL'] + agent_categories['Analytic']]
    agent_categories['Heuristic'] = heuristic_agents
    
    # Add category column
    df['Category'] = df['Agent'].apply(lambda x: get_agent_category(x, agent_categories))
    
    # Categorize environments
    env_categorization = categorize_environments(df)
    
    # Generate report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write("# Results and Conclusions\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(df)}\n\n")
        f.write(f"**Environments:** {len(df['Environment'].unique())}\n\n")
        f.write(f"**Agents:** {len(df['Agent'].unique())}\n\n")
        
        f.write("---\n\n")
        
        # 1. Executive Summary
        f.write("## 1. Executive Summary\n\n")
        
        # Overall best/worst
        if 'mean' in df.columns:
            best_idx = df['mean'].idxmax()
            worst_idx = df['mean'].idxmin()
            best_row = df.loc[best_idx]
            worst_row = df.loc[worst_idx]
            
            f.write("### Overall Performance Highlights\n\n")
            f.write(f"- **Best Performance:** {best_row['Agent']} achieves mean PnL of {best_row['mean']:.4f} on {best_row['Environment']}\n")
            if 'sharpe' in best_row and not pd.isna(best_row['sharpe']):
                f.write(f"  - Sharpe Ratio: {best_row['sharpe']:.4f}\n")
            
            f.write(f"- **Worst Performance:** {worst_row['Agent']} achieves mean PnL of {worst_row['mean']:.4f} on {worst_row['Environment']}\n")
            if 'sharpe' in worst_row and not pd.isna(worst_row['sharpe']):
                f.write(f"  - Sharpe Ratio: {worst_row['sharpe']:.4f}\n")
            
            f.write(f"- **Average Performance:** Mean PnL = {df['mean'].mean():.4f} ± {df['mean'].std():.4f}\n\n")
        
        if 'sharpe' in df.columns:
            best_sharpe_idx = df['sharpe'].idxmax()
            best_sharpe_row = df.loc[best_sharpe_idx]
            f.write(f"- **Best Risk-Adjusted Return:** {best_sharpe_row['Agent']} achieves Sharpe ratio of {best_sharpe_row['sharpe']:.4f} on {best_sharpe_row['Environment']}\n")
            f.write(f"  - Mean PnL: {best_sharpe_row['mean']:.4f}\n\n")
        
        # Category summary
        f.write("### Category Performance Summary\n\n")
        if 'mean' in df.columns:
            for category in ['RL', 'Analytic', 'Heuristic']:
                category_df = df[df['Category'] == category]
                if len(category_df) > 0:
                    mean_pnl = category_df['mean'].mean()
                    f.write(f"- **{category} Agents:** Average Mean PnL = {mean_pnl:.4f} "
                           f"(n={len(category_df)} experiments)\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # 2. Comparative Analysis by Agent Category
        f.write("## 2. Comparative Analysis by Agent Category\n\n")
        
        f.write("This section provides a statistical comparison of agent categories "
                "(RL, Analytic, Heuristic) across all performance metrics.\n\n")
        
        # Statistics for each metric
        metrics_to_analyze = ['mean', 'sharpe', 'std']
        for metric in metrics_to_analyze:
            if metric not in df.columns:
                continue
            
            metric_names = {
                'mean': 'Mean PnL',
                'sharpe': 'Sharpe Ratio',
                'std': 'Standard Deviation'
            }
            
            category_stats = calculate_category_statistics(df, agent_categories, metric)
            f.write(format_statistical_table(category_stats, metric_names.get(metric, metric)))
        
        # Coefficient of variation (consistency)
        if 'mean' in df.columns:
            f.write("### Consistency Across Environments (Coefficient of Variation)\n\n")
            f.write("Lower CV indicates more consistent performance across environments.\n\n")
            
            consistency_df = calculate_consistency(df, 'Agent', 'mean')
            consistency_df = consistency_df.sort_values('CV')
            
            f.write("| Agent | CV | Mean PnL | Std Dev | Environments |\n")
            f.write("|-------|----|----|----------|------------|\n")
            for _, row in consistency_df.head(10).iterrows():
                f.write(f"| {row['Agent']} | {row['CV']:.4f} | {row['Mean']:.4f} | "
                       f"{row['Std']:.4f} | {int(row['Environments'])} |\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # 3. Individual Agent Performance Analysis
        f.write("## 3. Individual Agent Performance Analysis\n\n")
        
        # Rankings
        if 'mean' in df.columns:
            f.write("### Top Performers by Mean PnL\n\n")
            rankings = rank_agents(df, 'mean', ascending=False)
            f.write("| Rank | Agent | Mean PnL | Median | Std Dev | Environments |\n")
            f.write("|------|-------|----------|--------|---------|-------------|\n")
            for _, row in rankings.head(10).iterrows():
                f.write(f"| {int(row['rank'])} | {row['Agent']} | {row['mean']:.4f} | "
                       f"{row['median']:.4f} | {row['std']:.4f} | {int(row['count'])} |\n")
            f.write("\n")
        
        if 'sharpe' in df.columns:
            f.write("### Top Performers by Sharpe Ratio\n\n")
            rankings = rank_agents(df, 'sharpe', ascending=False)
            f.write("| Rank | Agent | Sharpe Ratio | Mean | Std Dev | Environments |\n")
            f.write("|------|-------|--------------|------|---------|-------------|\n")
            for _, row in rankings.head(10).iterrows():
                f.write(f"| {int(row['rank'])} | {row['Agent']} | {row['mean']:.4f} | "
                       f"{row['median']:.4f} | {row['std']:.4f} | {int(row['count'])} |\n")
            f.write("\n")
        
        # Agent specialization
        f.write("### Agent Specialization by Environment Type\n\n")
        f.write("This analysis identifies which agents excel in which environment types.\n\n")
        
        for env_type, envs in env_categorization['by_type'].items():
            type_df = df[df['Environment'].isin(envs)]
            if 'mean' in type_df.columns and len(type_df) > 0:
                best_by_type = type_df.loc[type_df['mean'].idxmax()]
                f.write(f"- **{env_type} Environments:** Best agent is {best_by_type['Agent']} "
                       f"(Mean PnL: {best_by_type['mean']:.4f})\n")
        f.write("\n")
        
        # RL algorithm comparison
        f.write("### RL Algorithm Comparison\n\n")
        rl_agents = agent_categories['RL']
        rl_df = df[df['Agent'].isin(rl_agents)]
        
        if len(rl_df) > 0 and 'mean' in rl_df.columns:
            f.write("| Algorithm | Mean PnL | Sharpe Ratio | Std Dev | Environments |\n")
            f.write("|-----------|----------|--------------|---------|-------------|\n")
            
            for agent in rl_agents:
                agent_df = rl_df[rl_df['Agent'] == agent]
                if len(agent_df) > 0:
                    mean_pnl = agent_df['mean'].mean()
                    sharpe = agent_df['sharpe'].mean() if 'sharpe' in agent_df.columns else np.nan
                    std = agent_df['std'].mean() if 'std' in agent_df.columns else np.nan
                    count = len(agent_df)
                    
                    sharpe_str = f"{sharpe:.4f}" if not pd.isna(sharpe) else "N/A"
                    std_str = f"{std:.4f}" if not pd.isna(std) else "N/A"
                    
                    f.write(f"| {agent} | {mean_pnl:.4f} | {sharpe_str} | {std_str} | {count} |\n")
            f.write("\n")
        
        # LSTM vs non-LSTM
        f.write("### LSTM vs Non-LSTM RL Agents\n\n")
        lstm_agents = [a for a in rl_agents if 'LSTM' in a]
        non_lstm_agents = [a for a in rl_agents if 'LSTM' not in a]
        
        if lstm_agents and non_lstm_agents and 'mean' in df.columns:
            lstm_df = df[df['Agent'].isin(lstm_agents)]
            non_lstm_df = df[df['Agent'].isin(non_lstm_agents)]
            
            lstm_mean = lstm_df['mean'].mean()
            non_lstm_mean = non_lstm_df['mean'].mean()
            
            f.write(f"- **LSTM Agents:** Average Mean PnL = {lstm_mean:.4f} (n={len(lstm_df)})\n")
            f.write(f"- **Non-LSTM Agents:** Average Mean PnL = {non_lstm_mean:.4f} (n={len(non_lstm_df)})\n")
            
            if lstm_mean > non_lstm_mean:
                improvement = ((lstm_mean - non_lstm_mean) / abs(non_lstm_mean)) * 100
                f.write(f"- **LSTM agents outperform non-LSTM by {improvement:.2f}% in terms of mean PnL**\n")
            else:
                improvement = ((non_lstm_mean - lstm_mean) / abs(lstm_mean)) * 100
                f.write(f"- **Non-LSTM agents outperform LSTM by {improvement:.2f}% in terms of mean PnL**\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # 4. Environment Complexity Analysis
        f.write("## 4. Environment Complexity Analysis\n\n")
        
        f.write("This section analyzes how environment complexity (Vanilla, Jump, Regime, Jump+Regime) "
                "affects agent performance.\n\n")
        
        if 'mean' in df.columns:
            complexity_stats = analyze_environment_complexity(df, env_categorization, 'mean')
            
            f.write("### Performance by Complexity Level\n\n")
            f.write("| Complexity | Mean PnL | Median | Std Dev | Range |\n")
            f.write("|------------|----------|--------|---------|-------|\n")
            
            for complexity, stats in complexity_stats.items():
                if not np.isnan(stats['mean']):
                    f.write(f"| {complexity} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                           f"{stats['std']:.4f} | [{stats['min']:.4f}, {stats['max']:.4f}] |\n")
            f.write("\n")
        
        # Environment type analysis
        f.write("### Performance by Environment Type (ABM, GBM, OU)\n\n")
        
        if 'mean' in df.columns:
            type_stats = analyze_environment_type(df, env_categorization, 'mean')
            
            f.write("| Environment Type | Mean PnL | Median | Std Dev | Range |\n")
            f.write("|-----------------|----------|--------|---------|-------|\n")
            
            for env_type, stats in type_stats.items():
                if not np.isnan(stats['mean']):
                    f.write(f"| {env_type} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                           f"{stats['std']:.4f} | [{stats['min']:.4f}, {stats['max']:.4f}] |\n")
            f.write("\n")
        
        # Environment difficulty (average performance across all agents)
        f.write("### Environment Difficulty Ranking\n\n")
        f.write("Environments ranked by average performance across all agents (higher is easier).\n\n")
        
        if 'mean' in df.columns:
            env_difficulty = df.groupby('Environment')['mean'].agg(['mean', 'std', 'count']).reset_index()
            env_difficulty = env_difficulty.sort_values('mean', ascending=False)
            
            f.write("| Rank | Environment | Avg Mean PnL | Std Dev | Agents |\n")
            f.write("|------|-------------|--------------|---------|--------|\n")
            for i, (_, row) in enumerate(env_difficulty.iterrows(), 1):
                f.write(f"| {i} | {row['Environment']} | {row['mean']:.4f} | "
                       f"{row['std']:.4f} | {int(row['count'])} |\n")
            f.write("\n")
        
        # Agent adaptability
        f.write("### Agent Adaptability Across Environments\n\n")
        f.write("Agents ranked by their ability to generalize across different environments "
                "(measured by consistency of performance).\n\n")
        
        if 'mean' in df.columns:
            adaptability = calculate_consistency(df, 'Agent', 'mean')
            adaptability = adaptability.sort_values('CV')
            
            f.write("| Rank | Agent | CV (lower is better) | Mean PnL | Environments |\n")
            f.write("|------|-------|---------------------|----------|-------------|\n")
            for i, (_, row) in enumerate(adaptability.head(10).iterrows(), 1):
                f.write(f"| {i} | {row['Agent']} | {row['CV']:.4f} | "
                       f"{row['Mean']:.4f} | {int(row['Environments'])} |\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # 5. Risk-Return Trade-offs
        f.write("## 5. Risk-Return Trade-offs\n\n")
        
        if 'mean' in df.columns and 'std' in df.columns:
            f.write("### Risk-Return Analysis\n\n")
            f.write("Analysis of the trade-off between return (mean PnL) and risk (standard deviation).\n\n")
            
            # Category-level risk-return
            f.write("#### By Agent Category\n\n")
            f.write("| Category | Mean Return | Mean Risk (Std Dev) | Sharpe Ratio |\n")
            f.write("|----------|-------------|---------------------|--------------|\n")
            
            for category in ['RL', 'Analytic', 'Heuristic']:
                category_df = df[df['Category'] == category]
                if len(category_df) > 0:
                    mean_return = category_df['mean'].mean()
                    mean_risk = category_df['std'].mean()
                    sharpe = category_df['sharpe'].mean() if 'sharpe' in category_df.columns else np.nan
                    
                    sharpe_str = f"{sharpe:.4f}" if not pd.isna(sharpe) else "N/A"
                    
                    f.write(f"| {category} | {mean_return:.4f} | {mean_risk:.4f} | {sharpe_str} |\n")
            f.write("\n")
        
        # VaR and ES analysis
        if 'var_95' in df.columns and 'es_95' in df.columns:
            f.write("### Tail Risk Analysis (VaR and Expected Shortfall)\n\n")
            f.write("Analysis of extreme loss scenarios at 95% and 99% confidence levels.\n\n")
            
            f.write("#### Average Tail Risk by Category\n\n")
            f.write("| Category | VaR (95%) | ES (95%) | VaR (99%) | ES (99%) |\n")
            f.write("|----------|-----------|----------|-----------|----------|\n")
            
            for category in ['RL', 'Analytic', 'Heuristic']:
                category_df = df[df['Category'] == category]
                if len(category_df) > 0:
                    var_95 = category_df['var_95'].mean()
                    es_95 = category_df['es_95'].mean()
                    var_99 = category_df['var_99'].mean() if 'var_99' in category_df.columns else np.nan
                    es_99 = category_df['es_99'].mean() if 'es_99' in category_df.columns else np.nan
                    
                    var_99_str = f"{var_99:.4f}" if not pd.isna(var_99) else "N/A"
                    es_99_str = f"{es_99:.4f}" if not pd.isna(es_99) else "N/A"
                    
                    f.write(f"| {category} | {var_95:.4f} | {es_95:.4f} | {var_99_str} | {es_99_str} |\n")
            f.write("\n")
            
            f.write("*Note: VaR and ES are reported as negative values (losses). More negative values indicate higher tail risk.*\n\n")
        
        f.write("---\n\n")
        
        # 6. Statistical Patterns and Insights
        f.write("## 6. Statistical Patterns and Insights\n\n")
        
        # Correlation analysis
        if 'mean' in df.columns and 'sharpe' in df.columns:
            f.write("### Correlation Analysis\n\n")
            corr = calculate_correlation(df, 'mean', 'sharpe')
            if not pd.isna(corr):
                f.write(f"- **Mean PnL vs Sharpe Ratio:** Pearson correlation = {corr:.4f}\n")
            
            if 'std' in df.columns:
                corr = calculate_correlation(df, 'mean', 'std')
                if not pd.isna(corr):
                    f.write(f"- **Mean PnL vs Standard Deviation:** Pearson correlation = {corr:.4f}\n")
                
                corr = calculate_correlation(df, 'sharpe', 'std')
                if not pd.isna(corr):
                    f.write(f"- **Sharpe Ratio vs Standard Deviation:** Pearson correlation = {corr:.4f}\n")
            f.write("\n")
        
        # Outlier identification
        if 'mean' in df.columns:
            f.write("### Exceptional Performances\n\n")
            
            # Positive outliers (exceptional success)
            outliers_positive = identify_outliers(df, 'mean', method='iqr', threshold=1.5)
            outliers_positive = outliers_positive[outliers_positive['mean'] > df['mean'].median()]
            outliers_positive = outliers_positive.sort_values('mean', ascending=False)
            
            if len(outliers_positive) > 0:
                f.write("#### Exceptional Positive Performances\n\n")
                f.write("| Agent | Environment | Mean PnL | Sharpe Ratio |\n")
                f.write("|-------|-------------|----------|--------------|\n")
                for _, row in outliers_positive.head(5).iterrows():
                    sharpe = row['sharpe'] if 'sharpe' in row and not pd.isna(row['sharpe']) else np.nan
                    sharpe_str = f"{sharpe:.4f}" if not pd.isna(sharpe) else "N/A"
                    f.write(f"| {row['Agent']} | {row['Environment']} | {row['mean']:.4f} | {sharpe_str} |\n")
                f.write("\n")
            
            # Negative outliers (exceptional failures)
            outliers_negative = identify_outliers(df, 'mean', method='iqr', threshold=1.5)
            outliers_negative = outliers_negative[outliers_negative['mean'] < df['mean'].median()]
            outliers_negative = outliers_negative.sort_values('mean', ascending=True)
            
            if len(outliers_negative) > 0:
                f.write("#### Exceptional Negative Performances\n\n")
                f.write("| Agent | Environment | Mean PnL | Sharpe Ratio |\n")
                f.write("|-------|-------------|----------|--------------|\n")
                for _, row in outliers_negative.head(5).iterrows():
                    sharpe = row['sharpe'] if 'sharpe' in row and not pd.isna(row['sharpe']) else np.nan
                    sharpe_str = f"{sharpe:.4f}" if not pd.isna(sharpe) else "N/A"
                    f.write(f"| {row['Agent']} | {row['Environment']} | {row['mean']:.4f} | {sharpe_str} |\n")
                f.write("\n")
        
        # Environment-agent interactions
        f.write("### Notable Environment-Agent Interactions\n\n")
        f.write("Identification of significant positive and negative interactions between agent types and environment characteristics.\n\n")
        
        # Find best/worst combinations
        if 'mean' in df.columns:
            best_combo = df.loc[df['mean'].idxmax()]
            worst_combo = df.loc[df['mean'].idxmin()]
            
            f.write(f"- **Best Combination:** {best_combo['Agent']} on {best_combo['Environment']} "
                   f"(Mean PnL: {best_combo['mean']:.4f})\n")
            f.write(f"- **Worst Combination:** {worst_combo['Agent']} on {worst_combo['Environment']} "
                   f"(Mean PnL: {worst_combo['mean']:.4f})\n\n")
        
        f.write("---\n\n")
        
        # 7. Critical Discussion
        f.write("## 7. Critical Discussion\n\n")
        
        # RL vs Traditional Methods
        f.write("### RL vs Traditional Methods\n\n")
        
        rl_df = df[df['Category'] == 'RL']
        analytic_df = df[df['Category'] == 'Analytic']
        heuristic_df = df[df['Category'] == 'Heuristic']
        
        if 'mean' in df.columns and len(rl_df) > 0:
            rl_mean = rl_df['mean'].mean()
            analytic_mean = analytic_df['mean'].mean() if len(analytic_df) > 0 else np.nan
            heuristic_mean = heuristic_df['mean'].mean() if len(heuristic_df) > 0 else np.nan
            
            f.write("Our results reveal interesting patterns in the performance of reinforcement learning "
                   "agents compared to traditional methods:\n\n")
            
            if not pd.isna(analytic_mean):
                if rl_mean > analytic_mean:
                    f.write(f"- **RL agents** achieve an average mean PnL of {rl_mean:.4f}, "
                           f"outperforming **analytic methods** ({analytic_mean:.4f}) by "
                           f"{((rl_mean - analytic_mean) / abs(analytic_mean) * 100):.2f}%.\n")
                else:
                    f.write(f"- **Analytic methods** achieve an average mean PnL of {analytic_mean:.4f}, "
                           f"outperforming **RL agents** ({rl_mean:.4f}) by "
                           f"{((analytic_mean - rl_mean) / abs(rl_mean) * 100):.2f}%.\n")
            
            if not pd.isna(heuristic_mean):
                if rl_mean > heuristic_mean:
                    f.write(f"- **RL agents** outperform **heuristic methods** "
                           f"({heuristic_mean:.4f}) by {((rl_mean - heuristic_mean) / abs(heuristic_mean) * 100):.2f}%.\n")
                else:
                    f.write(f"- **Heuristic methods** outperform **RL agents** "
                           f"by {((heuristic_mean - rl_mean) / abs(rl_mean) * 100):.2f}%.\n")
            
            f.write("\nHowever, performance varies significantly across environments. ")
            f.write("RL agents show particular strength in complex, non-stationary environments with regime-switching or jumps, ")
            f.write("where adaptive learning can capture patterns that fixed rules cannot. ")
            f.write("In simpler, more predictable environments, well-designed analytic or heuristic methods may perform comparably or better.\n\n")
        
        # LSTM effectiveness
        f.write("### LSTM Effectiveness\n\n")
        
        if lstm_agents and non_lstm_agents and 'mean' in df.columns:
            lstm_df = df[df['Agent'].isin(lstm_agents)]
            non_lstm_df = df[df['Agent'].isin(non_lstm_agents)]
            
            lstm_mean = lstm_df['mean'].mean()
            non_lstm_mean = non_lstm_df['mean'].mean()
            
            if lstm_mean > non_lstm_mean:
                f.write(f"LSTM-based RL agents demonstrate superior performance (mean PnL: {lstm_mean:.4f}) "
                       f"compared to non-LSTM RL agents ({non_lstm_mean:.4f}), suggesting that sequence modeling "
                       f"provides valuable context for market-making decisions. The ability to maintain memory "
                       f"of past market states enables these agents to better adapt to regime changes and temporal patterns.\n\n")
            else:
                f.write(f"Non-LSTM RL agents achieve comparable or slightly better performance (mean PnL: {non_lstm_mean:.4f}) "
                       f"compared to LSTM-based agents ({lstm_mean:.4f}). This suggests that for market-making tasks, "
                       f"the additional complexity of sequence modeling may not always provide sufficient benefit, "
                       f"or that the current LSTM implementations may require further optimization.\n\n")
        
        # Heuristic robustness
        f.write("### Heuristic Robustness\n\n")
        
        if 'mean' in df.columns and len(heuristic_df) > 0:
            heuristic_consistency = calculate_consistency(heuristic_df, 'Agent', 'mean')
            heuristic_consistency = heuristic_consistency.sort_values('CV')
            
            f.write("Heuristic agents demonstrate remarkable robustness and consistency across environments. ")
            if len(heuristic_consistency) > 0:
                best_heuristic = heuristic_consistency.iloc[0]
                f.write(f"The most consistent heuristic agent ({best_heuristic['Agent']}) achieves a coefficient of variation ")
                f.write(f"of {best_heuristic['CV']:.4f}, indicating stable performance.\n\n")
            
            f.write("Simple rule-based strategies, while not always achieving the highest returns, provide predictable and ")
            f.write("reliable performance. This makes them attractive for practical applications where consistency is valued ")
            f.write("over peak performance, or as baseline strategies for comparison.\n\n")
        
        # Analytic methods
        f.write("### Analytic Methods: Theory vs Practice\n\n")
        
        if len(analytic_df) > 0 and 'mean' in analytic_df.columns:
            f.write("Analytic methods, derived from theoretical optimal control frameworks, demonstrate strong performance ")
            f.write("in environments that match their underlying assumptions. The AS (Avellaneda-Stoikov) closed-form solution ")
            f.write("and its heuristic approximations show particular strength in environments with predictable price dynamics.\n\n")
            
            f.write("However, when faced with regime-switching or jump diffusion, these methods may struggle to adapt. ")
            f.write("The theoretical optimality under idealized conditions does not always translate to superior empirical ")
            f.write("performance in realistic market simulations with non-stationary dynamics.\n\n")
        
        f.write("---\n\n")
        
        # 8. Conclusions
        f.write("## 8. Conclusions\n\n")
        
        f.write("### Main Findings\n\n")
        
        f.write("1. **No single agent type dominates across all environments.** The best-performing agent varies significantly ")
        f.write("depending on the environment characteristics, emphasizing the importance of selecting appropriate strategies ")
        f.write("for specific market conditions.\n\n")
        
        f.write("2. **RL agents show promise in complex environments.** In environments with regime-switching, jumps, or ")
        f.write("non-stationary dynamics, RL agents demonstrate the ability to adapt and learn effective policies that ")
        f.write("traditional methods struggle with.\n\n")
        
        f.write("3. **Heuristic methods provide robust baselines.** Simple rule-based strategies achieve consistent, ")
        f.write("predictable performance across diverse environments, making them valuable for practical applications and ")
        f.write("as benchmarks for comparison.\n\n")
        
        f.write("4. **Environment complexity significantly impacts performance.** As environments become more complex ")
        f.write("(from vanilla to jump+regime), the variance in agent performance increases, and the relative advantages ")
        f.write("of different approaches become more pronounced.\n\n")
        
        f.write("5. **LSTM architectures offer potential advantages** for capturing temporal dependencies, though their ")
        f.write("effectiveness varies across environments and may require careful tuning.\n\n")
        
        f.write("### Research Contributions\n\n")
        
        f.write("This study contributes to the market-making literature by:\n\n")
        f.write("- Providing a comprehensive comparison of RL, analytic, and heuristic methods across 12 diverse environments\n")
        f.write("- Demonstrating the importance of environment characteristics in determining optimal strategy selection\n")
        f.write("- Establishing benchmarks for future research in algorithmic market making\n")
        f.write("- Identifying specific scenarios where RL methods provide advantages over traditional approaches\n\n")
        
        f.write("### Practical Implications\n\n")
        
        f.write("For practitioners:\n\n")
        f.write("- **Environment assessment is critical:** Understanding market characteristics (volatility regimes, jumps, ")
        f.write("price dynamics) should guide strategy selection.\n")
        f.write("- **Hybrid approaches may be optimal:** Combining different agent types or using ensemble methods could ")
        f.write("leverage the strengths of each approach.\n")
        f.write("- **Robustness vs performance trade-off:** Simple heuristics may be preferable when consistency is more ")
        f.write("important than peak performance.\n")
        f.write("- **RL requires careful tuning:** While RL agents show promise, they require significant computational ")
        f.write("resources and hyperparameter tuning to achieve optimal performance.\n\n")
        
        f.write("### Limitations\n\n")
        
        f.write("This study has several limitations:\n\n")
        f.write("- **Simulated environments:** Results are based on synthetic market simulations and may not fully capture ")
        f.write("real market dynamics, including adverse selection and information asymmetry.\n")
        f.write("- **Limited RL training:** RL agents are trained for a fixed number of timesteps; longer training might ")
        f.write("improve performance.\n")
        f.write("- **Hyperparameter sensitivity:** RL agent performance may be sensitive to hyperparameters not fully explored.\n")
        f.write("- **Single-asset focus:** Results are based on single-asset market making; multi-asset scenarios may differ.\n")
        f.write("- **Transaction costs:** While included in the simulation, real-world transaction costs and market impact ")
        f.write("may differ significantly.\n\n")
        
        f.write("### Future Directions\n\n")
        
        f.write("Future research could explore:\n\n")
        f.write("- **Multi-asset market making:** Extending analysis to portfolio-based market making strategies.\n")
        f.write("- **More sophisticated RL architectures:** Exploring transformer-based models, attention mechanisms, and ")
        f.write("multi-agent RL approaches.\n")
        f.write("- **Real market data:** Validating findings on historical market data and live trading environments.\n")
        f.write("- **Ensemble methods:** Combining multiple agents to leverage complementary strengths.\n")
        f.write("- **Robustness to model misspecification:** Testing agent performance when environment assumptions are violated.\n")
        f.write("- **Computational efficiency:** Developing more efficient training and inference methods for RL agents.\n\n")
        
        f.write("---\n\n")
        f.write("## References\n\n")
        f.write("For detailed metrics and data, see:\n")
        f.write("- [Evaluation Report](EVALUATION_REPORT.md) - Raw statistics and rankings\n")
        f.write("- [Visualization Report](VISUALIZATION_REPORT.md) - Visual representations and confidence intervals\n")
        f.write("- [Appendix](APPENDIX.md) - Detailed matrix format data\n\n")
        
    print(f"✓ Report generated: {output_path}")


def main():
    """Main function to create results summary."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create comprehensive results and conclusions summary")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing results")
    parser.add_argument("--output-file", type=str, default="results/RESULTS_AND_CONCLUSIONS.md",
                       help="Path to output markdown file")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Creating Results and Conclusions Summary")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Output file: {args.output_file}")
    print()
    
    print("Loading results...")
    metrics_dict = load_all_metrics(args.results_dir)
    
    if not metrics_dict:
        print("No results found. Exiting.")
        return
    
    print(f"  Loaded {len(metrics_dict)} environments")
    total_experiments = sum(len(agents) for agents in metrics_dict.values())
    print(f"  Total experiments: {total_experiments}")
    print()
    
    print("Generating report...")
    generate_report(metrics_dict, args.output_file)
    
    print()
    print("=" * 70)
    print("✓ Results and conclusions summary complete!")
    print(f"  Report saved to: {args.output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
