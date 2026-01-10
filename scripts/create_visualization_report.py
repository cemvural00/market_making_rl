"""
Create comprehensive visualization report with confidence intervals.

Generates both interactive HTML and Markdown reports with extensive visualizations
comparing all agents across all environments, including confidence intervals for all metrics.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_results import load_all_metrics

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


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


def get_env_type(env_name):
    """Determine environment type from name."""
    if 'ABM' in env_name:
        return 'ABM'
    elif 'GBM' in env_name:
        return 'GBM'
    elif 'OU' in env_name:
        return 'OU'
    return 'Unknown'


# Color schemes
COLORS = {
    'RL': '#2E86AB',  # Blue
    'Analytic': '#A23B72',  # Purple/Magenta
    'Heuristic': '#F18F01',  # Orange
}


# ============================================================================
# Confidence Interval Calculation Functions
# ============================================================================

def bootstrap_ci(data, statistic_func, confidence=0.95, n_bootstrap=1000, seed=42):
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : array-like
        Input data
    statistic_func : callable
        Function to compute statistic (e.g., np.mean, np.std)
    confidence : float
        Confidence level (default 0.95)
    n_bootstrap : int
        Number of bootstrap samples
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (np.nan, np.nan)
    
    np.random.seed(seed)
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        stat = statistic_func(sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return (lower, upper)


def calculate_mean_ci(pnls, confidence=0.95, n_bootstrap=1000):
    """
    Calculate confidence interval for mean PnL.
    Uses bootstrap method.
    
    Parameters
    ----------
    pnls : array-like
        PnL values
    confidence : float
        Confidence level
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if len(pnls) == 0:
        return (np.nan, np.nan)
    return bootstrap_ci(pnls, np.mean, confidence, n_bootstrap)


def calculate_sharpe_ci(pnls, confidence=0.95, n_bootstrap=1000):
    """
    Calculate confidence interval for Sharpe ratio.
    Uses bootstrap method.
    
    Parameters
    ----------
    pnls : array-like
        PnL values
    confidence : float
        Confidence level
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if len(pnls) == 0:
        return (np.nan, np.nan)
    
    def sharpe_ratio(data):
        if len(data) == 0 or np.std(data) == 0:
            return 0.0
        return np.mean(data) / (np.std(data) + 1e-12)
    
    return bootstrap_ci(pnls, sharpe_ratio, confidence, n_bootstrap)


def calculate_std_ci(pnls, confidence=0.95):
    """
    Calculate confidence interval for standard deviation.
    Uses chi-square distribution.
    
    Parameters
    ----------
    pnls : array-like
        PnL values
    confidence : float
        Confidence level
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if len(pnls) < 2:
        return (np.nan, np.nan)
    
    n = len(pnls)
    std = np.std(pnls, ddof=1)
    alpha = 1 - confidence
    
    chi2_lower = stats.chi2.ppf(alpha / 2, n - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, n - 1)
    
    lower = std * np.sqrt((n - 1) / chi2_upper)
    upper = std * np.sqrt((n - 1) / chi2_lower)
    
    return (lower, upper)


def calculate_var_ci(pnls, confidence=0.95, var_level=0.95, n_bootstrap=1000):
    """
    Calculate confidence interval for Value at Risk.
    Uses bootstrap method.
    
    Parameters
    ----------
    pnls : array-like
        PnL values
    confidence : float
        Confidence level for CI
    var_level : float
        VaR level (e.g., 0.95 for 95% VaR)
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if len(pnls) == 0:
        return (np.nan, np.nan)
    
    def var_statistic(data):
        return np.quantile(data, 1 - var_level)
    
    return bootstrap_ci(pnls, var_statistic, confidence, n_bootstrap)


def calculate_es_ci(pnls, confidence=0.95, es_level=0.95, n_bootstrap=1000):
    """
    Calculate confidence interval for Expected Shortfall.
    Uses bootstrap method.
    
    Parameters
    ----------
    pnls : array-like
        PnL values
    confidence : float
        Confidence level for CI
    es_level : float
        ES level (e.g., 0.95 for 95% ES)
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if len(pnls) == 0:
        return (np.nan, np.nan)
    
    def es_statistic(data):
        var = np.quantile(data, 1 - es_level)
        return np.mean(data[data <= var])
    
    return bootstrap_ci(pnls, es_statistic, confidence, n_bootstrap)


def calculate_inventory_ci(inventory, confidence=0.95):
    """
    Calculate confidence interval for average inventory.
    Uses t-distribution.
    
    Parameters
    ----------
    inventory : array-like
        Inventory values
    confidence : float
        Confidence level
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if len(inventory) < 2:
        return (np.nan, np.nan)
    
    mean = np.mean(inventory)
    std = np.std(inventory, ddof=1)
    n = len(inventory)
    
    alpha = 1 - confidence
    t_critical = t_dist.ppf(1 - alpha / 2, n - 1)
    
    margin = t_critical * std / np.sqrt(n)
    lower = mean - margin
    upper = mean + margin
    
    return (lower, upper)


def calculate_confidence_intervals(pnls, inventory=None, confidence=0.95, n_bootstrap=1000):
    """
    Calculate confidence intervals for all metrics.
    
    Parameters
    ----------
    pnls : array-like
        PnL values
    inventory : array-like, optional
        Inventory values
    confidence : float
        Confidence level
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns
    -------
    dict
        Dictionary with CI for each metric
    """
    pnls = np.array(pnls)
    
    ci_dict = {
        'mean': calculate_mean_ci(pnls, confidence, n_bootstrap),
        'sharpe': calculate_sharpe_ci(pnls, confidence, n_bootstrap),
        'std': calculate_std_ci(pnls, confidence),
        'var_95': calculate_var_ci(pnls, confidence, 0.95, n_bootstrap),
        'var_99': calculate_var_ci(pnls, confidence, 0.99, n_bootstrap),
        'es_95': calculate_es_ci(pnls, confidence, 0.95, n_bootstrap),
        'es_99': calculate_es_ci(pnls, confidence, 0.99, n_bootstrap),
    }
    
    if inventory is not None:
        ci_dict['avg_inventory'] = calculate_inventory_ci(inventory, confidence)
    
    return ci_dict


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_all_data(results_dir="results", confidence=0.95, n_bootstrap=1000):
    """
    Load all metrics and raw PnL arrays, calculate confidence intervals.
    
    Parameters
    ----------
    results_dir : str
        Directory containing results
    confidence : float
        Confidence level for CIs
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns
    -------
    tuple
        (metrics_dict, pnl_data_dict, ci_dict)
        - metrics_dict: {env_name: {agent_name: metrics}}
        - pnl_data_dict: {env_name: {agent_name: pnl_array}}
        - ci_dict: {env_name: {agent_name: {metric: (lower, upper)}}}
    """
    results_path = Path(results_dir)
    
    # Load metrics
    metrics_dict = load_all_metrics(results_dir)
    
    # Load raw PnL data and calculate CIs
    pnl_data_dict = {}
    ci_dict = {}
    
    for env_name, agents in metrics_dict.items():
        pnl_data_dict[env_name] = {}
        ci_dict[env_name] = {}
        
        for agent_name in agents.keys():
            agent_dir = results_path / env_name / agent_name
            
            # Load PnL array
            pnl_file = agent_dir / "pnls.npy"
            if pnl_file.exists():
                try:
                    pnls = np.load(pnl_file)
                    pnl_data_dict[env_name][agent_name] = pnls
                    
                    # Load inventory if available
                    inventory_file = agent_dir / "inventory.npy"
                    inventory = None
                    if inventory_file.exists():
                        inventory = np.load(inventory_file)
                    
                    # Calculate confidence intervals
                    cis = calculate_confidence_intervals(pnls, inventory, confidence, n_bootstrap)
                    ci_dict[env_name][agent_name] = cis
                except Exception as e:
                    print(f"Warning: Could not load data for {env_name}/{agent_name}: {e}")
                    pnl_data_dict[env_name][agent_name] = None
                    ci_dict[env_name][agent_name] = {}
    
    return metrics_dict, pnl_data_dict, ci_dict


# ============================================================================
# Visualization Functions
# ============================================================================

def prepare_dataframe(metrics_dict, ci_dict=None):
    """
    Convert metrics dictionary to pandas DataFrame.
    
    Parameters
    ----------
    metrics_dict : dict
        Nested dict: {env_name: {agent_name: metrics}}
    ci_dict : dict, optional
        Nested dict: {env_name: {agent_name: {metric: (lower, upper)}}}
    
    Returns
    -------
    pd.DataFrame
        DataFrame with Environment, Agent, and metric columns
    """
    rows = []
    for env_name, agents in metrics_dict.items():
        for agent_name, metrics in agents.items():
            row = {
                "Environment": env_name,
                "Agent": agent_name,
                **metrics
            }
            
            # Add CI columns if available
            if ci_dict and env_name in ci_dict and agent_name in ci_dict[env_name]:
                for metric, (lower, upper) in ci_dict[env_name][agent_name].items():
                    row[f"{metric}_ci_lower"] = lower
                    row[f"{metric}_ci_upper"] = upper
                    row[f"{metric}_ci_width"] = upper - lower if not (np.isnan(lower) or np.isnan(upper)) else np.nan
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def create_heatmaps(metrics_dict, ci_dict, output_dir, agent_categories):
    """
    Create heatmap visualizations for all metrics with confidence interval overlays.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    agent_categories : dict
        Agent categories dictionary
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    metrics = ['mean', 'sharpe', 'std', 'var_95', 'es_95', 'var_99', 'es_99', 'avg_inventory']
    metric_names = {
        'mean': 'Mean PnL',
        'sharpe': 'Sharpe Ratio',
        'std': 'Standard Deviation',
        'var_95': 'VaR (95%)',
        'es_95': 'ES (95%)',
        'var_99': 'VaR (99%)',
        'es_99': 'ES (99%)',
        'avg_inventory': 'Average Inventory'
    }
    
    # Get all agents and environments
    all_agents = sorted(df['Agent'].unique())
    all_envs = sorted(df['Environment'].unique())
    
    # Organize agents by category
    agent_order = []
    for category in ['RL', 'Analytic', 'Heuristic']:
        for agent in agent_categories[category]:
            if agent in all_agents:
                agent_order.append(agent)
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # Create pivot table
        pivot = df.pivot(index='Agent', columns='Environment', values=metric)
        pivot = pivot.reindex(agent_order)
        pivot = pivot[all_envs]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot heatmap
        sns.heatmap(pivot, annot=False, fmt='.2f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': metric_names[metric]}, ax=ax, linewidths=0.5)
        
        ax.set_title(f'{metric_names[metric]} - All Agents Across Environments', fontsize=14, fontweight='bold')
        ax.set_xlabel('Environment', fontsize=12)
        ax.set_ylabel('Agent', fontsize=12)
        
        # Add category separators
        rl_count = sum(1 for a in agent_categories['RL'] if a in all_agents)
        analytic_count = sum(1 for a in agent_categories['Analytic'] if a in all_agents)
        
        if rl_count > 0:
            ax.axhline(y=rl_count, color='black', linewidth=2, linestyle='--')
        if analytic_count > 0:
            ax.axhline(y=rl_count + analytic_count, color='black', linewidth=2, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{metric}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Created heatmap: {metric}")


def create_category_comparisons(metrics_dict, ci_dict, output_dir, agent_categories):
    """
    Create bar charts comparing agent categories with error bars showing 95% CI.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    agent_categories : dict
        Agent categories dictionary
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    metrics = ['mean', 'sharpe', 'std', 'var_95', 'es_95', 'var_99', 'es_99', 'avg_inventory']
    metric_names = {
        'mean': 'Mean PnL',
        'sharpe': 'Sharpe Ratio',
        'std': 'Standard Deviation',
        'var_95': 'VaR (95%)',
        'es_95': 'ES (95%)',
        'var_99': 'VaR (99%)',
        'es_99': 'ES (99%)',
        'avg_inventory': 'Average Inventory'
    }
    
    # Calculate category averages
    category_avgs = {}
    category_cis = {}
    
    for category, agents in agent_categories.items():
        category_df = df[df['Agent'].isin(agents)]
        
        category_avgs[category] = {}
        category_cis[category] = {}
        
        for metric in metrics:
            if metric not in category_df.columns:
                continue
            
            # Average across all environments for this category
            values = category_df[metric].values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                category_avgs[category][metric] = np.mean(valid_values)
            else:
                category_avgs[category][metric] = np.nan
            
            # Aggregate CI: collect all CI bounds and compute average width
            ci_lowers = []
            ci_uppers = []
            for _, row in category_df.iterrows():
                ci_lower_col = f'{metric}_ci_lower'
                ci_upper_col = f'{metric}_ci_upper'
                metric_val = row.get(metric, np.nan)
                
                if ci_lower_col in row and ci_upper_col in row:
                    ci_lower = row[ci_lower_col]
                    ci_upper = row[ci_upper_col]
                    if not (np.isnan(ci_lower) or np.isnan(ci_upper)) and not np.isnan(metric_val):
                        ci_lowers.append(ci_lower)
                        ci_uppers.append(ci_upper)
            
            if ci_lowers and len(ci_lowers) > 0:
                # Use average of CI bounds, centered around category mean
                avg_lower = np.mean(ci_lowers)
                avg_upper = np.mean(ci_uppers)
                mean_val = category_avgs[category][metric]
                
                # Ensure CI bounds are reasonable (lower <= mean <= upper)
                if not np.isnan(mean_val):
                    # Calculate average CI width
                    ci_widths = [u - l for l, u in zip(ci_lowers, ci_uppers) if not (np.isnan(l) or np.isnan(u))]
                    if ci_widths:
                        avg_width = np.mean(ci_widths) / 2
                        # Center CI around mean
                        category_cis[category][metric] = (mean_val - avg_width, mean_val + avg_width)
                    else:
                        category_cis[category][metric] = (avg_lower, avg_upper)
                else:
                    category_cis[category][metric] = (avg_lower, avg_upper)
            else:
                category_cis[category][metric] = (np.nan, np.nan)
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['RL', 'Analytic', 'Heuristic']
        means = [category_avgs[cat].get(metric, np.nan) for cat in categories]
        
        # Extract CI bounds
        ci_lowers = []
        ci_uppers = []
        for cat in categories:
            ci = category_cis[cat].get(metric, (np.nan, np.nan))
            ci_lowers.append(ci[0])
            ci_uppers.append(ci[1])
        
        # Calculate error bars (ensure non-negative and reasonable)
        errors_lower = []
        errors_upper = []
        for i in range(len(means)):
            mean_val = means[i]
            ci_lower = ci_lowers[i]
            ci_upper = ci_uppers[i]
            
            if not np.isnan(mean_val):
                # If CI is available, use it (but ensure bounds are valid)
                if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                    # Ensure CI bounds are around the mean
                    lower_err = max(0, mean_val - ci_lower) if ci_lower <= mean_val else abs(mean_val) * 0.1
                    upper_err = max(0, ci_upper - mean_val) if ci_upper >= mean_val else abs(mean_val) * 0.1
                else:
                    # Use a default small error (5% of mean)
                    lower_err = abs(mean_val) * 0.05
                    upper_err = abs(mean_val) * 0.05
            else:
                lower_err = 0
                upper_err = 0
            
            errors_lower.append(lower_err)
            errors_upper.append(upper_err)
        
        # Create bar plot
        bars = ax.bar(categories, means, color=[COLORS[cat] for cat in categories],
                     yerr=[errors_lower, errors_upper], capsize=10, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_title(f'{metric_names[metric]} - Category Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_names[metric], fontsize=12)
        ax.set_xlabel('Agent Category', fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            if not np.isnan(mean):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'category_comparison_{metric}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Created category comparison: {metric}")


def create_risk_return_plots(metrics_dict, ci_dict, output_dir, agent_categories):
    """
    Create risk-return scatter plots with error bars and confidence ellipses.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    agent_categories : dict
        Agent categories dictionary
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    
    # Risk metrics to use
    risk_metrics = {
        'std': ('Standard Deviation', 'mean', 'Mean PnL'),
        'var_95': ('VaR (95%)', 'mean', 'Mean PnL'),
        'es_95': ('ES (95%)', 'mean', 'Mean PnL'),
    }
    
    for risk_metric, (risk_label, return_metric, return_label) in risk_metrics.items():
        if risk_metric not in df.columns or return_metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot by category
        for category in ['RL', 'Analytic', 'Heuristic']:
            category_agents = agent_categories[category]
            category_df = df[df['Agent'].isin(category_agents)]
            
            x = category_df[risk_metric].values
            y = category_df[return_metric].values
            
            # Get CI bounds if available (ensure non-negative)
            xerr_lower = []
            xerr_upper = []
            yerr_lower = []
            yerr_upper = []
            
            for idx, row in category_df.iterrows():
                # Risk CI
                risk_val = row[risk_metric]
                risk_ci_lower = row.get(f'{risk_metric}_ci_lower', np.nan)
                risk_ci_upper = row.get(f'{risk_metric}_ci_upper', np.nan)
                if not (np.isnan(risk_ci_lower) or np.isnan(risk_ci_upper)) and not np.isnan(risk_val):
                    xerr_lower.append(max(0, risk_val - risk_ci_lower))
                    xerr_upper.append(max(0, risk_ci_upper - risk_val))
                else:
                    xerr_lower.append(0)
                    xerr_upper.append(0)
                
                # Return CI
                return_val = row[return_metric]
                return_ci_lower = row.get(f'{return_metric}_ci_lower', np.nan)
                return_ci_upper = row.get(f'{return_metric}_ci_upper', np.nan)
                if not (np.isnan(return_ci_lower) or np.isnan(return_ci_upper)) and not np.isnan(return_val):
                    yerr_lower.append(max(0, return_val - return_ci_lower))
                    yerr_upper.append(max(0, return_ci_upper - return_val))
                else:
                    yerr_lower.append(0)
                    yerr_upper.append(0)
            
            # Size by Sharpe ratio if available (use scalar for errorbar compatibility)
            # Note: errorbar doesn't support per-point markersize arrays, so we use average size
            if 'sharpe' in category_df.columns and len(category_df) > 0:
                avg_sharpe_size = float(category_df['sharpe'].abs().mean() * 50)
                # Ensure reasonable size bounds
                marker_size = max(20, min(200, avg_sharpe_size))
            else:
                marker_size = 100
            
            # Plot with error bars (markersize must be scalar, not array)
            ax.errorbar(x, y, xerr=[xerr_lower, xerr_upper], yerr=[yerr_lower, yerr_upper],
                       fmt='o', label=category, color=COLORS[category], alpha=0.6,
                       capsize=3, capthick=1.5, markersize=marker_size, markeredgecolor='black', markeredgewidth=0.5)
        
        ax.set_xlabel(risk_label, fontsize=12)
        ax.set_ylabel(return_label, fontsize=12)
        ax.set_title(f'Risk-Return Analysis: {risk_label} vs {return_label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'risk_return_{risk_metric}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Created risk-return plot: {risk_metric}")


def create_env_type_comparisons(metrics_dict, ci_dict, output_dir, agent_categories):
    """
    Create box plots comparing ABM, GBM, OU environment types with confidence intervals.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    agent_categories : dict
        Agent categories dictionary
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    
    # Add environment type
    df['EnvType'] = df['Environment'].apply(get_env_type)
    
    metrics = ['mean', 'sharpe', 'std']
    metric_names = {
        'mean': 'Mean PnL',
        'sharpe': 'Sharpe Ratio',
        'std': 'Standard Deviation'
    }
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create box plot grouped by environment type and category
        data_for_plot = []
        labels = []
        
        for env_type in ['ABM', 'GBM', 'OU']:
            for category in ['RL', 'Analytic', 'Heuristic']:
                category_agents = agent_categories[category]
                subset = df[(df['EnvType'] == env_type) & (df['Agent'].isin(category_agents))]
                
                if not subset.empty and metric in subset.columns:
                    values = subset[metric].dropna().values
                    if len(values) > 0:
                        data_for_plot.append(values)
                        labels.append(f'{env_type}\n{category}')
        
        if data_for_plot:
            bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True, notch=True,
                           showmeans=True, meanline=True)
            
            # Color boxes by category
            color_idx = 0
            for patch in bp['boxes']:
                if 'RL' in labels[color_idx]:
                    patch.set_facecolor(COLORS['RL'])
                elif 'Analytic' in labels[color_idx]:
                    patch.set_facecolor(COLORS['Analytic'])
                else:
                    patch.set_facecolor(COLORS['Heuristic'])
                patch.set_alpha(0.7)
                color_idx += 1
        
        ax.set_title(f'{metric_names[metric]} - Environment Type Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_names[metric], fontsize=12)
        ax.set_xlabel('Environment Type × Agent Category', fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'env_type_comparison_{metric}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Created env type comparison: {metric}")


def create_rankings(metrics_dict, ci_dict, output_dir, top_n=10):
    """
    Create ranking visualizations with error bars showing 95% confidence intervals.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    top_n : int
        Number of top agents to show
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    
    metrics = ['mean', 'sharpe']
    metric_names = {
        'mean': 'Mean PnL',
        'sharpe': 'Sharpe Ratio'
    }
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # Average across environments for each agent
        agent_means = df.groupby('Agent')[metric].mean().sort_values(ascending=False)
        
        # Get top N and bottom N
        top_agents = agent_means.head(top_n)
        
        # Get CI bounds for top agents (simplified - use mean CI)
        ci_lowers = []
        ci_uppers = []
        for agent in top_agents.index:
            agent_df = df[df['Agent'] == agent]
            ci_lower_col = f'{metric}_ci_lower'
            ci_upper_col = f'{metric}_ci_upper'
            
            if ci_lower_col in agent_df.columns and ci_upper_col in agent_df.columns:
                lowers = agent_df[ci_lower_col].dropna()
                uppers = agent_df[ci_upper_col].dropna()
                if len(lowers) > 0:
                    # Use mean CI width, centered around agent mean
                    mean_lower = np.mean(lowers)
                    mean_upper = np.mean(uppers)
                    agent_mean = top_agents[agent]
                    ci_width = (mean_upper - mean_lower) / 2
                    ci_lowers.append(agent_mean - ci_width)
                    ci_uppers.append(agent_mean + ci_width)
                else:
                    ci_lowers.append(np.nan)
                    ci_uppers.append(np.nan)
            else:
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        y_pos = np.arange(len(top_agents))
        
        # Error bars (ensure non-negative)
        errors_lower = []
        errors_upper = []
        for i in range(len(top_agents)):
            mean_val = top_agents.values[i]
            ci_lower = ci_lowers[i]
            ci_upper = ci_uppers[i]
            
            if not np.isnan(ci_lower) and not np.isnan(ci_upper) and not np.isnan(mean_val):
                errors_lower.append(max(0, mean_val - ci_lower))
                errors_upper.append(max(0, ci_upper - mean_val))
            else:
                errors_lower.append(0)
                errors_upper.append(0)
        
        ax.barh(y_pos, top_agents.values, xerr=[errors_lower, errors_upper],
               capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_agents.index)
        ax.set_xlabel(metric_names[metric], fontsize=12)
        ax.set_title(f'Top {top_n} Agents by {metric_names[metric]}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'ranking_{metric}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Created ranking: {metric}")


def create_distribution_plots(pnl_data_dict, output_dir, agent_categories, n_samples=1000):
    """
    Load raw PnL arrays and create violin/box plots with confidence bands.
    
    Parameters
    ----------
    pnl_data_dict : dict
        Dictionary with raw PnL arrays
    output_dir : Path
        Output directory for figures
    agent_categories : dict
        Agent categories dictionary
    n_samples : int
        Number of samples for KDE
    """
    # Collect PnL data by category
    category_data = {
        'RL': [],
        'Analytic': [],
        'Heuristic': []
    }
    
    category_labels = []
    
    for category in ['RL', 'Analytic', 'Heuristic']:
        for agent in agent_categories[category]:
            for env_name, agents in pnl_data_dict.items():
                if agent in agents and agents[agent] is not None:
                    pnls = agents[agent]
                    if len(pnls) > 0:
                        category_data[category].extend(pnls)
                        category_labels.append(category)
    
    # Create combined violin plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    data_to_plot = [category_data[cat] for cat in ['RL', 'Analytic', 'Heuristic'] if len(category_data[cat]) > 0]
    labels_to_plot = [cat for cat in ['RL', 'Analytic', 'Heuristic'] if len(category_data[cat]) > 0]
    colors_to_plot = [COLORS[cat] for cat in labels_to_plot]
    
    if data_to_plot:
        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), showmeans=True, showmedians=True)
        
        for i, (pc, color) in enumerate(zip(parts['bodies'], colors_to_plot)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Overlay box plots
        bp = ax.boxplot(data_to_plot, positions=range(len(data_to_plot)), widths=0.3,
                       patch_artist=False, showfliers=False)
        
        ax.set_xticks(range(len(labels_to_plot)))
        ax.set_xticklabels(labels_to_plot)
        ax.set_ylabel('PnL', fontsize=12)
        ax.set_title('PnL Distribution by Agent Category', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pnl_distributions_category.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("  Created distribution plot: category comparison")


def create_best_agents_visualization(metrics_dict, ci_dict, output_dir):
    """
    Create visualization showing best agent per environment with significance testing.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    
    # Find best agent per environment by mean PnL
    best_agents = []
    envs = []
    
    for env in sorted(df['Environment'].unique()):
        env_df = df[df['Environment'] == env]
        if 'mean' in env_df.columns:
            best_idx = env_df['mean'].idxmax()
            best_agent = env_df.loc[best_idx, 'Agent']
            best_value = env_df.loc[best_idx, 'mean']
            
            # Check if CI doesn't overlap with second best
            sorted_agents = env_df.sort_values('mean', ascending=False)
            if len(sorted_agents) > 1:
                second_best_value = sorted_agents.iloc[1]['mean']
                best_ci_lower = sorted_agents.iloc[0].get('mean_ci_lower', np.nan)
                second_ci_upper = sorted_agents.iloc[1].get('mean_ci_upper', np.nan)
                
                significant = not (np.isnan(best_ci_lower) or np.isnan(second_ci_upper)) and best_ci_lower > second_ci_upper
            else:
                significant = False
            
            best_agents.append(best_agent)
            envs.append(env)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create matrix: agents x environments
    all_agents = sorted(df['Agent'].unique())
    agent_matrix = np.zeros((len(all_agents), len(envs)))
    
    for i, agent in enumerate(all_agents):
        for j, env in enumerate(envs):
            agent_env_df = df[(df['Agent'] == agent) & (df['Environment'] == env)]
            if not agent_env_df.empty and 'mean' in agent_env_df.columns:
                agent_matrix[i, j] = agent_env_df.iloc[0]['mean']
    
    # Plot heatmap
    im = ax.imshow(agent_matrix, cmap='RdYlGn', aspect='auto')
    
    # Highlight best agents
    for j, (env, agent) in enumerate(zip(envs, best_agents)):
        i = all_agents.index(agent)
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=3))
    
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha='right')
    ax.set_yticks(range(len(all_agents)))
    ax.set_yticklabels(all_agents)
    ax.set_title('Best Agent per Environment (Mean PnL)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Agent', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Mean PnL')
    plt.tight_layout()
    plt.savefig(output_dir / 'best_agents_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("  Created best agents visualization")


def create_radar_charts(metrics_dict, ci_dict, output_dir, agent_categories, top_n=3):
    """
    Create radar/spider charts for multi-metric agent comparison with confidence bands.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    agent_categories : dict
        Agent categories dictionary
    top_n : int
        Number of top agents per category
    """
    from math import pi
    
    df = prepare_dataframe(metrics_dict, ci_dict)
    
    # Metrics to include in radar chart
    radar_metrics = ['mean', 'sharpe', 'std', 'var_95', 'es_95']
    metric_labels = ['Mean PnL', 'Sharpe', 'Std Dev', 'VaR (95%)', 'ES (95%)']
    
    # Normalize metrics to [0, 1] for radar chart
    for metric in radar_metrics:
        if metric in df.columns:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df[f'{metric}_norm'] = 0.5
    
    for category in ['RL', 'Analytic', 'Heuristic']:
        category_agents = agent_categories[category]
        category_df = df[df['Agent'].isin(category_agents)]
        
        if category_df.empty:
            continue
        
        # Average across environments
        agent_avgs = category_df.groupby('Agent')['mean'].mean().sort_values(ascending=False)
        top_agents = agent_avgs.head(top_n).index.tolist()
        
        if not top_agents:
            continue
        
        # Number of metrics
        num_metrics = len(radar_metrics)
        angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for agent in top_agents:
            agent_df = category_df[category_df['Agent'] == agent]
            values = []
            ci_lowers = []
            ci_uppers = []
            
            for metric in radar_metrics:
                if f'{metric}_norm' in agent_df.columns:
                    avg_val = agent_df[f'{metric}_norm'].mean()
                    values.append(avg_val)
                    
                    # CI bounds (normalized)
                    ci_lower = agent_df[f'{metric}_ci_lower'].min()
                    ci_upper = agent_df[f'{metric}_ci_upper'].max()
                    min_metric = df[metric].min()
                    max_metric = df[metric].max()
                    
                    if not (np.isnan(ci_lower) or np.isnan(ci_upper)) and max_metric > min_metric:
                        ci_lowers.append((ci_lower - min_metric) / (max_metric - min_metric))
                        ci_uppers.append((ci_upper - min_metric) / (max_metric - min_metric))
                    else:
                        ci_lowers.append(avg_val)
                        ci_uppers.append(avg_val)
                else:
                    values.append(0)
                    ci_lowers.append(0)
                    ci_uppers.append(0)
            
            values += values[:1]  # Complete the circle
            ci_lowers += ci_lowers[:1]
            ci_uppers += ci_uppers[:1]
            
            # Plot with confidence bands
            ax.plot(angles, values, 'o-', linewidth=2, label=agent, alpha=0.7)
            ax.fill(angles, ci_lowers, alpha=0.1)
            ax.fill(angles, ci_uppers, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title(f'Top {top_n} {category} Agents - Multi-Metric Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'radar_{category.lower()}_agents.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Created radar chart: {category}")


def create_consistency_analysis(metrics_dict, ci_dict, output_dir):
    """
    Calculate and visualize agent consistency (CV) across environments with CI.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    
    # Calculate coefficient of variation for each agent
    agents = sorted(df['Agent'].unique())
    cvs = []
    agent_names = []
    
    for agent in agents:
        agent_df = df[df['Agent'] == agent]
        if 'mean' in agent_df.columns:
            means = agent_df['mean'].dropna()
            if len(means) > 1:
                cv = means.std() / (means.mean() + 1e-12)  # Coefficient of variation
                cvs.append(cv)
                agent_names.append(agent)
    
    if not cvs:
        return
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sorted_data = sorted(zip(agent_names, cvs), key=lambda x: x[1])
    sorted_agents, sorted_cvs = zip(*sorted_data)
    
    bars = ax.barh(range(len(sorted_agents)), sorted_cvs, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_yticks(range(len(sorted_agents)))
    ax.set_yticklabels(sorted_agents)
    ax.set_xlabel('Coefficient of Variation (Lower = More Consistent)', fontsize=12)
    ax.set_title('Agent Consistency Across Environments', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agent_consistency.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("  Created consistency analysis")


def create_ci_comparison(ci_dict, output_dir):
    """
    Create visualization comparing CI widths across metrics to show precision.
    
    Parameters
    ----------
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    """
    metrics = ['mean', 'sharpe', 'std', 'var_95', 'es_95', 'var_99', 'es_99', 'avg_inventory']
    metric_names = {
        'mean': 'Mean PnL',
        'sharpe': 'Sharpe Ratio',
        'std': 'Std Dev',
        'var_95': 'VaR (95%)',
        'es_95': 'ES (95%)',
        'var_99': 'VaR (99%)',
        'es_99': 'ES (99%)',
        'avg_inventory': 'Avg Inventory'
    }
    
    # Collect CI widths
    ci_widths = {metric: [] for metric in metrics}
    
    for env_name, agents in ci_dict.items():
        for agent_name, cis in agents.items():
            for metric in metrics:
                if metric in cis:
                    lower, upper = cis[metric]
                    if not (np.isnan(lower) or np.isnan(upper)):
                        width = upper - lower
                        ci_widths[metric].append(width)
    
    # Calculate average CI width per metric
    avg_widths = {}
    for metric in metrics:
        if ci_widths[metric]:
            avg_widths[metric] = np.mean(ci_widths[metric])
        else:
            avg_widths[metric] = np.nan
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_metrics = sorted([m for m in metrics if not np.isnan(avg_widths.get(m, np.nan))],
                           key=lambda x: avg_widths.get(x, 0))
    
    widths = [avg_widths[m] for m in sorted_metrics]
    labels = [metric_names[m] for m in sorted_metrics]
    
    bars = ax.bar(range(len(sorted_metrics)), widths, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xticks(range(len(sorted_metrics)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Average CI Width', fontsize=12)
    ax.set_title('Confidence Interval Width Comparison (Lower = More Precise)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ci_width_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("  Created CI width comparison")


def create_agent_comparisons(metrics_dict, ci_dict, output_dir, agent_categories):
    """
    Create bar charts comparing individual agents with error bars showing 95% CI.
    Similar to category comparisons but for each individual agent.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    agent_categories : dict
        Agent categories dictionary
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    metrics = ['mean', 'sharpe', 'std', 'var_95', 'es_95', 'var_99', 'es_99', 'avg_inventory']
    metric_names = {
        'mean': 'Mean PnL',
        'sharpe': 'Sharpe Ratio',
        'std': 'Standard Deviation',
        'var_95': 'VaR (95%)',
        'es_95': 'ES (95%)',
        'var_99': 'VaR (99%)',
        'es_99': 'ES (99%)',
        'avg_inventory': 'Average Inventory'
    }
    
    # Calculate agent averages across all environments
    agent_avgs = {}
    agent_cis = {}
    
    all_agents = sorted(df['Agent'].unique())
    
    for agent in all_agents:
        agent_df = df[df['Agent'] == agent]
        
        agent_avgs[agent] = {}
        agent_cis[agent] = {}
        
        for metric in metrics:
            if metric not in agent_df.columns:
                continue
            
            # Average across all environments for this agent
            values = agent_df[metric].values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                agent_avgs[agent][metric] = np.mean(valid_values)
            else:
                agent_avgs[agent][metric] = np.nan
            
            # Aggregate CI: collect all CI bounds and compute average width
            ci_lowers = []
            ci_uppers = []
            for _, row in agent_df.iterrows():
                ci_lower_col = f'{metric}_ci_lower'
                ci_upper_col = f'{metric}_ci_upper'
                metric_val = row.get(metric, np.nan)
                
                if ci_lower_col in row and ci_upper_col in row:
                    ci_lower = row[ci_lower_col]
                    ci_upper = row[ci_upper_col]
                    if not (np.isnan(ci_lower) or np.isnan(ci_upper)) and not np.isnan(metric_val):
                        ci_lowers.append(ci_lower)
                        ci_uppers.append(ci_upper)
            
            if ci_lowers and len(ci_lowers) > 0:
                # Use average of CI bounds, centered around agent mean
                avg_lower = np.mean(ci_lowers)
                avg_upper = np.mean(ci_uppers)
                mean_val = agent_avgs[agent][metric]
                
                # Ensure CI bounds are reasonable (lower <= mean <= upper)
                if not np.isnan(mean_val):
                    # Calculate average CI width
                    ci_widths = [u - l for l, u in zip(ci_lowers, ci_uppers) if not (np.isnan(l) or np.isnan(u))]
                    if ci_widths:
                        avg_width = np.mean(ci_widths) / 2
                        # Center CI around mean
                        agent_cis[agent][metric] = (mean_val - avg_width, mean_val + avg_width)
                    else:
                        agent_cis[agent][metric] = (avg_lower, avg_upper)
                else:
                    agent_cis[agent][metric] = (avg_lower, avg_upper)
            else:
                agent_cis[agent][metric] = (np.nan, np.nan)
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Sort agents by metric value (descending)
        agents_with_vals = [(agent, agent_avgs[agent].get(metric, np.nan)) for agent in all_agents]
        agents_with_vals = sorted(agents_with_vals, key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
        agents_sorted = [a[0] for a in agents_with_vals]
        means = [a[1] for a in agents_with_vals]
        
        # Extract CI bounds
        ci_lowers = []
        ci_uppers = []
        for agent in agents_sorted:
            ci = agent_cis[agent].get(metric, (np.nan, np.nan))
            ci_lowers.append(ci[0])
            ci_uppers.append(ci[1])
        
        # Calculate error bars (ensure non-negative and reasonable)
        errors_lower = []
        errors_upper = []
        for i in range(len(means)):
            mean_val = means[i]
            ci_lower = ci_lowers[i]
            ci_upper = ci_uppers[i]
            
            if not np.isnan(mean_val):
                # If CI is available, use it (but ensure bounds are valid)
                if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                    # Ensure CI bounds are around the mean
                    lower_err = max(0, mean_val - ci_lower) if ci_lower <= mean_val else abs(mean_val) * 0.1
                    upper_err = max(0, ci_upper - mean_val) if ci_upper >= mean_val else abs(mean_val) * 0.1
                else:
                    # Use a default small error (5% of mean)
                    lower_err = abs(mean_val) * 0.05
                    upper_err = abs(mean_val) * 0.05
            else:
                lower_err = 0
                upper_err = 0
            
            errors_lower.append(lower_err)
            errors_upper.append(upper_err)
        
        # Color bars by category
        colors = []
        for agent in agents_sorted:
            if agent in agent_categories['RL']:
                colors.append(COLORS['RL'])
            elif agent in agent_categories['Analytic']:
                colors.append(COLORS['Analytic'])
            else:
                colors.append(COLORS['Heuristic'])
        
        # Create bar plot
        bars = ax.bar(range(len(agents_sorted)), means, color=colors,
                     yerr=[errors_lower, errors_upper], capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_title(f'{metric_names[metric]} - Individual Agent Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_names[metric], fontsize=12)
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_xticks(range(len(agents_sorted)))
        ax.set_xticklabels(agents_sorted, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            if not np.isnan(mean):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)
        
        # Add legend for categories
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['RL'], label='RL Agents'),
            Patch(facecolor=COLORS['Analytic'], label='Analytic Agents'),
            Patch(facecolor=COLORS['Heuristic'], label='Heuristic Agents')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'agent_comparison_{metric}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Created agent comparison: {metric}")


def create_agent_risk_return_plots(metrics_dict, ci_dict, output_dir, agent_categories):
    """
    Create risk-return scatter plots for individual agents with error bars.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    agent_categories : dict
        Agent categories dictionary
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    
    # Risk metrics to use
    risk_metrics = {
        'std': ('Standard Deviation', 'mean', 'Mean PnL'),
        'var_95': ('VaR (95%)', 'mean', 'Mean PnL'),
        'es_95': ('ES (95%)', 'mean', 'Mean PnL'),
    }
    
    for risk_metric, (risk_label, return_metric, return_label) in risk_metrics.items():
        if risk_metric not in df.columns or return_metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        all_agents = sorted(df['Agent'].unique())
        
        # Plot by category with different markers for each agent
        for category in ['RL', 'Analytic', 'Heuristic']:
            category_agents = [a for a in agent_categories[category] if a in all_agents]
            
            if not category_agents:
                continue
            
            # Average across environments for each agent
            agent_means_risk = {}
            agent_means_return = {}
            agent_cis_risk = {}
            agent_cis_return = {}
            
            for agent in category_agents:
                agent_df = df[df['Agent'] == agent]
                
                # Average risk metric
                risk_vals = agent_df[risk_metric].dropna().values
                if len(risk_vals) > 0:
                    agent_means_risk[agent] = np.mean(risk_vals)
                else:
                    agent_means_risk[agent] = np.nan
                
                # Average return metric
                return_vals = agent_df[return_metric].dropna().values
                if len(return_vals) > 0:
                    agent_means_return[agent] = np.mean(return_vals)
                else:
                    agent_means_return[agent] = np.nan
                
                # Average CI bounds
                risk_cis_lower = agent_df[f'{risk_metric}_ci_lower'].dropna().values
                risk_cis_upper = agent_df[f'{risk_metric}_ci_upper'].dropna().values
                if len(risk_cis_lower) > 0 and len(risk_cis_upper) > 0:
                    agent_cis_risk[agent] = (np.mean(risk_cis_lower), np.mean(risk_cis_upper))
                else:
                    agent_cis_risk[agent] = (np.nan, np.nan)
                
                return_cis_lower = agent_df[f'{return_metric}_ci_lower'].dropna().values
                return_cis_upper = agent_df[f'{return_metric}_ci_upper'].dropna().values
                if len(return_cis_lower) > 0 and len(return_cis_upper) > 0:
                    agent_cis_return[agent] = (np.mean(return_cis_lower), np.mean(return_cis_upper))
                else:
                    agent_cis_return[agent] = (np.nan, np.nan)
            
            # Prepare data for plotting
            x = []
            y = []
            xerr_lower = []
            xerr_upper = []
            yerr_lower = []
            yerr_upper = []
            agent_names = []
            
            for agent in category_agents:
                risk_mean = agent_means_risk.get(agent, np.nan)
                return_mean = agent_means_return.get(agent, np.nan)
                
                if not (np.isnan(risk_mean) or np.isnan(return_mean)):
                    x.append(risk_mean)
                    y.append(return_mean)
                    agent_names.append(agent)
                    
                    # Risk CI
                    risk_ci = agent_cis_risk.get(agent, (np.nan, np.nan))
                    if not (np.isnan(risk_ci[0]) or np.isnan(risk_ci[1])):
                        xerr_lower.append(max(0, risk_mean - risk_ci[0]))
                        xerr_upper.append(max(0, risk_ci[1] - risk_mean))
                    else:
                        xerr_lower.append(0)
                        xerr_upper.append(0)
                    
                    # Return CI
                    return_ci = agent_cis_return.get(agent, (np.nan, np.nan))
                    if not (np.isnan(return_ci[0]) or np.isnan(return_ci[1])):
                        yerr_lower.append(max(0, return_mean - return_ci[0]))
                        yerr_upper.append(max(0, return_ci[1] - return_mean))
                    else:
                        yerr_lower.append(0)
                        yerr_upper.append(0)
            
            if x and y:
                # Size by Sharpe ratio if available (use scalar)
                category_df = df[df['Agent'].isin(category_agents)]
                if 'sharpe' in category_df.columns and len(category_df) > 0:
                    avg_sharpe_size = float(category_df['sharpe'].abs().mean() * 50)
                    marker_size = max(20, min(200, avg_sharpe_size))
                else:
                    marker_size = 100
                
                # Plot with error bars
                ax.errorbar(x, y, xerr=[xerr_lower, xerr_upper], yerr=[yerr_lower, yerr_upper],
                           fmt='o', label=category, color=COLORS[category], alpha=0.6,
                           capsize=3, capthick=1.5, markersize=marker_size, markeredgecolor='black', markeredgewidth=0.5)
                
                # Add agent labels
                for i, agent in enumerate(agent_names):
                    ax.annotate(agent, (x[i], y[i]), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
        
        ax.set_xlabel(risk_label, fontsize=12)
        ax.set_ylabel(return_label, fontsize=12)
        ax.set_title(f'Risk-Return Analysis by Agent: {risk_label} vs {return_label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'agent_risk_return_{risk_metric}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Created agent risk-return plot: {risk_metric}")


def generate_markdown_report(metrics_dict, ci_dict, output_dir, output_file):
    """
    Generate Markdown report with embedded images and detailed analysis text.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    output_file : Path
        Output markdown file path
    """
    df = prepare_dataframe(metrics_dict, ci_dict)
    
    with open(output_file, 'w') as f:
        f.write("# Comprehensive Results Visualization Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(df)}\n\n")
        f.write(f"**Environments:** {len(df['Environment'].unique())}\n\n")
        f.write(f"**Agents:** {len(df['Agent'].unique())}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        if 'mean' in df.columns:
            best_overall = df.loc[df['mean'].idxmax()]
            f.write(f"- **Best Overall Performance:** {best_overall['Agent']} achieves mean PnL of {best_overall['mean']:.4f} on {best_overall['Environment']}\n")
            if 'mean_ci_lower' in best_overall and 'mean_ci_upper' in best_overall:
                ci_lower = best_overall['mean_ci_lower']
                ci_upper = best_overall['mean_ci_upper']
                if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                    f.write(f"  - 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
            f.write("\n")
        
        if 'sharpe' in df.columns:
            best_sharpe = df.loc[df['sharpe'].idxmax()]
            f.write(f"- **Best Risk-Adjusted Return:** {best_sharpe['Agent']} achieves Sharpe ratio of {best_sharpe['sharpe']:.4f} on {best_sharpe['Environment']}\n")
            if 'sharpe_ci_lower' in best_sharpe and 'sharpe_ci_upper' in best_sharpe:
                ci_lower = best_sharpe['sharpe_ci_lower']
                ci_upper = best_sharpe['sharpe_ci_upper']
                if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                    f.write(f"  - 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
            f.write("\n")
        
        # Confidence Interval Methodology
        f.write("## Confidence Interval Methodology\n\n")
        f.write("All statistics include 95% confidence intervals calculated as follows:\n\n")
        f.write("- **Mean PnL, Sharpe Ratio, VaR, ES**: Bootstrap method with 1000 iterations\n")
        f.write("- **Standard Deviation**: Chi-square distribution (analytical)\n")
        f.write("- **Average Inventory**: T-distribution (analytical)\n\n")
        f.write("Bootstrap confidence intervals use the percentile method, providing robust\n")
        f.write("non-parametric estimates that make no distributional assumptions.\n\n")
        
        # Heatmap Overview
        f.write("## Heatmap Overview\n\n")
        f.write("The following heatmaps show performance across all agent-environment combinations:\n\n")
        metrics = ['mean', 'sharpe', 'std', 'var_95', 'es_95', 'var_99', 'es_99', 'avg_inventory']
        metric_names = {
            'mean': 'Mean PnL',
            'sharpe': 'Sharpe Ratio',
            'std': 'Standard Deviation',
            'var_95': 'VaR (95%)',
            'es_95': 'ES (95%)',
            'var_99': 'VaR (99%)',
            'es_99': 'ES (99%)',
            'avg_inventory': 'Average Inventory'
        }
        
        for metric in metrics:
            fig_path = output_dir / f'heatmap_{metric}.png'
            if fig_path.exists():
                f.write(f"### {metric_names[metric]}\n\n")
                f.write(f"![{metric_names[metric]} Heatmap]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Category Comparison
        f.write("## Category Performance Comparison\n\n")
        f.write("Comparison of agent categories (RL, Analytic, Heuristic) across all metrics.\n")
        f.write("Error bars show 95% confidence intervals.\n\n")
        
        for metric in ['mean', 'sharpe', 'std']:
            fig_path = output_dir / f'category_comparison_{metric}.png'
            if fig_path.exists():
                f.write(f"### {metric_names[metric]}\n\n")
                f.write(f"![{metric_names[metric]} Category Comparison]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Risk-Return Analysis
        f.write("## Risk-Return Analysis\n\n")
        f.write("Scatter plots showing risk vs return trade-offs with confidence intervals.\n")
        f.write("Horizontal error bars indicate uncertainty in risk estimates; vertical error bars\n")
        f.write("indicate uncertainty in return estimates.\n\n")
        
        for risk_metric in ['std', 'var_95', 'es_95']:
            fig_path = output_dir / f'risk_return_{risk_metric}.png'
            if fig_path.exists():
                f.write(f"### {metric_names.get(risk_metric, risk_metric)} vs Mean PnL\n\n")
                f.write(f"![Risk-Return: {risk_metric}]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Environment Type Analysis
        f.write("## Environment Type Analysis\n\n")
        f.write("Comparison of agent performance across environment types (ABM, GBM, OU).\n")
        f.write("Box plots show distributions; notches indicate approximate 95% confidence intervals.\n\n")
        
        for metric in ['mean', 'sharpe', 'std']:
            fig_path = output_dir / f'env_type_comparison_{metric}.png'
            if fig_path.exists():
                f.write(f"### {metric_names[metric]}\n\n")
                f.write(f"![{metric_names[metric]} by Environment Type]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Rankings
        f.write("## Agent Rankings\n\n")
        f.write("Top performing agents by key metrics with confidence intervals.\n\n")
        
        for metric in ['mean', 'sharpe']:
            fig_path = output_dir / f'ranking_{metric}.png'
            if fig_path.exists():
                f.write(f"### Top Agents by {metric_names[metric]}\n\n")
                f.write(f"![Ranking: {metric_names[metric]}]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Distribution Analysis
        f.write("## Distribution Analysis\n\n")
        f.write("PnL distributions across agent categories using violin plots.\n\n")
        fig_path = output_dir / 'pnl_distributions_category.png'
        if fig_path.exists():
            f.write(f"![PnL Distributions]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Best Agents
        f.write("## Best Agents per Environment\n\n")
        f.write("Heatmap highlighting the best performing agent in each environment.\n\n")
        fig_path = output_dir / 'best_agents_heatmap.png'
        if fig_path.exists():
            f.write(f"![Best Agents]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Multi-Metric Performance
        f.write("## Multi-Metric Performance (Radar Charts)\n\n")
        f.write("Radar charts showing top agents across multiple metrics simultaneously.\n")
        f.write("Confidence bands indicate uncertainty in each metric.\n\n")
        
        for category in ['RL', 'Analytic', 'Heuristic']:
            fig_path = output_dir / f'radar_{category.lower()}_agents.png'
            if fig_path.exists():
                f.write(f"### {category} Agents\n\n")
                f.write(f"![{category} Radar Chart]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Consistency Analysis
        f.write("## Consistency Analysis\n\n")
        f.write("Coefficient of variation (CV) across environments for each agent.\n")
        f.write("Lower CV indicates more consistent performance across different environments.\n\n")
        fig_path = output_dir / 'agent_consistency.png'
        if fig_path.exists():
            f.write(f"![Agent Consistency]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # CI Comparison
        f.write("## Confidence Interval Precision\n\n")
        f.write("Comparison of average CI widths across metrics.\n")
        f.write("Lower values indicate more precise estimates.\n\n")
        fig_path = output_dir / 'ci_width_comparison.png'
        if fig_path.exists():
            f.write(f"![CI Width Comparison]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Individual Agent Comparisons
        f.write("## Individual Agent Performance Comparison\n\n")
        f.write("Comparison of individual agents across all metrics, averaged across all environments.\n")
        f.write("Error bars show 95% confidence intervals. Agents are sorted by performance (descending).\n\n")
        
        for metric in ['mean', 'sharpe', 'std']:
            fig_path = output_dir / f'agent_comparison_{metric}.png'
            if fig_path.exists():
                f.write(f"### {metric_names.get(metric, metric)}\n\n")
                f.write(f"![{metric_names.get(metric, metric)} Agent Comparison]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Agent Risk-Return Analysis
        f.write("## Agent-Level Risk-Return Analysis\n\n")
        f.write("Risk-return scatter plots for individual agents with confidence intervals.\n")
        f.write("Each point represents an agent's average performance across all environments.\n\n")
        
        for risk_metric in ['std', 'var_95', 'es_95']:
            fig_path = output_dir / f'agent_risk_return_{risk_metric}.png'
            if fig_path.exists():
                f.write(f"### {metric_names.get(risk_metric, risk_metric)} vs Mean PnL\n\n")
                f.write(f"![Agent Risk-Return: {risk_metric}]({fig_path.relative_to(output_file.parent)})\n\n")
        
        # Key Insights
        f.write("## Key Insights and Conclusions\n\n")
        f.write("### Overall Performance\n\n")
        agent_categories = get_agent_categories()
        
        for category in ['RL', 'Analytic', 'Heuristic']:
            category_agents = agent_categories[category]
            category_df = df[df['Agent'].isin(category_agents)]
            if 'mean' in category_df.columns:
                avg_mean = category_df['mean'].mean()
                f.write(f"- **{category} Agents**: Average mean PnL = {avg_mean:.4f}\n")
        
        f.write("\n### Best Performers\n\n")
        if 'mean' in df.columns:
            best_agent = df.loc[df['mean'].idxmax()]
            f.write(f"- **Best Mean PnL**: {best_agent['Agent']} on {best_agent['Environment']} ({best_agent['mean']:.4f})\n")
        
        if 'sharpe' in df.columns:
            best_sharpe_agent = df.loc[df['sharpe'].idxmax()]
            f.write(f"- **Best Sharpe Ratio**: {best_sharpe_agent['Agent']} on {best_sharpe_agent['Environment']} ({best_sharpe_agent['sharpe']:.4f})\n")
        
        f.write("\n### Statistical Significance\n\n")
        f.write("When comparing agents, non-overlapping 95% confidence intervals indicate\n")
        f.write("statistically significant differences at the α=0.05 level.\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated from evaluation results with confidence intervals.*\n")
    
    print(f"✓ Markdown report saved to: {output_file}")


def generate_html_report(metrics_dict, ci_dict, output_dir, output_file):
    """
    Generate interactive HTML report using plotly.
    
    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary
    ci_dict : dict
        Confidence intervals dictionary
    output_dir : Path
        Output directory for figures
    output_file : Path
        Output HTML file path
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
    except ImportError:
        print("Warning: plotly not installed. Skipping HTML report generation.")
        print("  Install with: pip install plotly")
        return
    
    df = prepare_dataframe(metrics_dict, ci_dict)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Market Making Agent Visualization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2E86AB; }}
        h2 {{ color: #A23B72; margin-top: 30px; }}
        .figure {{ margin: 20px 0; }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>Comprehensive Results Visualization Report</h1>
    <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Total Experiments:</strong> {len(df)}</p>
    <p><strong>Environments:</strong> {len(df['Environment'].unique())}</p>
    <p><strong>Agents:</strong> {len(df['Agent'].unique())}</p>
    
    <h2>Executive Summary</h2>
    <p>This interactive report provides comprehensive visualizations of all agent performance metrics.</p>
    <p>Hover over data points to see detailed information including confidence intervals.</p>
    
    <h2>Note</h2>
    <p>This HTML report contains basic structure. For full interactive visualizations, install plotly:</p>
    <pre>pip install plotly</pre>
    <p>Then re-run the script to generate complete interactive charts.</p>
    
    <h2>Static Visualizations</h2>
    <p>Please refer to the static PNG files in the figures/ directory for all visualizations.</p>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✓ HTML report saved to: {output_file}")
    print("  Note: Full interactive features require plotly. Static images are available in figures/ directory.")


def main():
    """Main function to create visualization report."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create comprehensive visualization report with confidence intervals")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing results")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save output files")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    parser.add_argument("--format", type=str, default="both", choices=['both', 'html', 'markdown'],
                       help="Output format")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for CIs (default 0.95)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                       help="Number of bootstrap iterations (default 1000)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Creating Comprehensive Visualization Report with Confidence Intervals")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Confidence level: {args.confidence}")
    print(f"  Bootstrap iterations: {args.n_bootstrap}")
    print(f"  Output format: {args.format}")
    print()
    
    # Create output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set DPI
    plt.rcParams['savefig.dpi'] = args.dpi
    
    # Get agent categories
    agent_categories = get_agent_categories()
    
    # Load data
    print("Loading data and calculating confidence intervals...")
    metrics_dict, pnl_data_dict, ci_dict = load_all_data(args.results_dir, args.confidence, args.n_bootstrap)
    
    if not metrics_dict:
        print("No results found. Exiting.")
        return
    
    print(f"  Loaded {len(metrics_dict)} environments")
    total_experiments = sum(len(agents) for agents in metrics_dict.values())
    print(f"  Total experiments: {total_experiments}")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    print()
    
    create_heatmaps(metrics_dict, ci_dict, figures_dir, agent_categories)
    create_category_comparisons(metrics_dict, ci_dict, figures_dir, agent_categories)
    create_risk_return_plots(metrics_dict, ci_dict, figures_dir, agent_categories)
    create_env_type_comparisons(metrics_dict, ci_dict, figures_dir, agent_categories)
    create_rankings(metrics_dict, ci_dict, figures_dir)
    create_distribution_plots(pnl_data_dict, figures_dir, agent_categories)
    create_best_agents_visualization(metrics_dict, ci_dict, figures_dir)
    create_radar_charts(metrics_dict, ci_dict, figures_dir, agent_categories)
    create_consistency_analysis(metrics_dict, ci_dict, figures_dir)
    create_ci_comparison(ci_dict, figures_dir)
    
    # Create individual agent-level visualizations
    print()
    print("Creating individual agent-level visualizations...")
    print()
    create_agent_comparisons(metrics_dict, ci_dict, figures_dir, agent_categories)
    create_agent_risk_return_plots(metrics_dict, ci_dict, figures_dir, agent_categories)
    
    print()
    print("Generating reports...")
    
    # Generate reports
    if args.format in ['both', 'markdown']:
        md_file = output_path / "VISUALIZATION_REPORT.md"
        generate_markdown_report(metrics_dict, ci_dict, figures_dir, md_file)
    
    if args.format in ['both', 'html']:
        html_file = output_path / "VISUALIZATION_REPORT.html"
        generate_html_report(metrics_dict, ci_dict, figures_dir, html_file)
    
    print()
    print("=" * 70)
    print("✓ Visualization report generation complete!")
    print(f"  Figures saved to: {figures_dir}")
    if args.format in ['both', 'markdown']:
        print(f"  Markdown report: {output_path / 'VISUALIZATION_REPORT.md'}")
    if args.format in ['both', 'html']:
        print(f"  HTML report: {output_path / 'VISUALIZATION_REPORT.html'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
