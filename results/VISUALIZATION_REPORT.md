# Comprehensive Results Visualization Report

**Generated:** 2026-01-10 15:37:07

**Total Experiments:** 204

**Environments:** 12

**Agents:** 17

## Executive Summary

- **Best Overall Performance:** FixedSpreadAgent achieves mean PnL of 10.0936 on ABMJumpRegimeEnv
  - 95% CI: [4.2553, 16.0455]

- **Best Risk-Adjusted Return:** DeepPPOAgent achieves Sharpe ratio of 0.3479 on ABMRegimeEnv
  - 95% CI: [0.1709, 0.5401]

## Confidence Interval Methodology

All statistics include 95% confidence intervals calculated as follows:

- **Mean PnL, Sharpe Ratio, VaR, ES**: Bootstrap method with 1000 iterations
- **Standard Deviation**: Chi-square distribution (analytical)
- **Average Inventory**: T-distribution (analytical)

Bootstrap confidence intervals use the percentile method, providing robust
non-parametric estimates that make no distributional assumptions.

## Heatmap Overview

The following heatmaps show performance across all agent-environment combinations:

### Mean PnL

![Mean PnL Heatmap](figures/heatmap_mean.png)

### Sharpe Ratio

![Sharpe Ratio Heatmap](figures/heatmap_sharpe.png)

### Standard Deviation

![Standard Deviation Heatmap](figures/heatmap_std.png)

### VaR (95%)

![VaR (95%) Heatmap](figures/heatmap_var_95.png)

### ES (95%)

![ES (95%) Heatmap](figures/heatmap_es_95.png)

### VaR (99%)

![VaR (99%) Heatmap](figures/heatmap_var_99.png)

### ES (99%)

![ES (99%) Heatmap](figures/heatmap_es_99.png)

### Average Inventory

![Average Inventory Heatmap](figures/heatmap_avg_inventory.png)

## Category Performance Comparison

Comparison of agent categories (RL, Analytic, Heuristic) across all metrics.
Error bars show 95% confidence intervals.

### Mean PnL

![Mean PnL Category Comparison](figures/category_comparison_mean.png)

### Sharpe Ratio

![Sharpe Ratio Category Comparison](figures/category_comparison_sharpe.png)

### Standard Deviation

![Standard Deviation Category Comparison](figures/category_comparison_std.png)

## Risk-Return Analysis

Scatter plots showing risk vs return trade-offs with confidence intervals.
Horizontal error bars indicate uncertainty in risk estimates; vertical error bars
indicate uncertainty in return estimates.

### Standard Deviation vs Mean PnL

![Risk-Return: std](figures/risk_return_std.png)

### VaR (95%) vs Mean PnL

![Risk-Return: var_95](figures/risk_return_var_95.png)

### ES (95%) vs Mean PnL

![Risk-Return: es_95](figures/risk_return_es_95.png)

## Environment Type Analysis

Comparison of agent performance across environment types (ABM, GBM, OU).
Box plots show distributions; notches indicate approximate 95% confidence intervals.

### Mean PnL

![Mean PnL by Environment Type](figures/env_type_comparison_mean.png)

### Sharpe Ratio

![Sharpe Ratio by Environment Type](figures/env_type_comparison_sharpe.png)

### Standard Deviation

![Standard Deviation by Environment Type](figures/env_type_comparison_std.png)

## Agent Rankings

Top performing agents by key metrics with confidence intervals.

### Top Agents by Mean PnL

![Ranking: Mean PnL](figures/ranking_mean.png)

### Top Agents by Sharpe Ratio

![Ranking: Sharpe Ratio](figures/ranking_sharpe.png)

## Distribution Analysis

PnL distributions across agent categories using violin plots.

![PnL Distributions](figures/pnl_distributions_category.png)

## Best Agents per Environment

Heatmap highlighting the best performing agent in each environment.

![Best Agents](figures/best_agents_heatmap.png)

## Multi-Metric Performance (Radar Charts)

Radar charts showing top agents across multiple metrics simultaneously.
Confidence bands indicate uncertainty in each metric.

### RL Agents

![RL Radar Chart](figures/radar_rl_agents.png)

### Analytic Agents

![Analytic Radar Chart](figures/radar_analytic_agents.png)

### Heuristic Agents

![Heuristic Radar Chart](figures/radar_heuristic_agents.png)

## Consistency Analysis

Coefficient of variation (CV) across environments for each agent.
Lower CV indicates more consistent performance across different environments.

![Agent Consistency](figures/agent_consistency.png)

## Confidence Interval Precision

Comparison of average CI widths across metrics.
Lower values indicate more precise estimates.

![CI Width Comparison](figures/ci_width_comparison.png)

## Key Insights and Conclusions

### Overall Performance

- **RL Agents**: Average mean PnL = 2.0827
- **Analytic Agents**: Average mean PnL = 2.6700
- **Heuristic Agents**: Average mean PnL = 2.4562

### Best Performers

- **Best Mean PnL**: FixedSpreadAgent on ABMJumpRegimeEnv (10.0936)
- **Best Sharpe Ratio**: DeepPPOAgent on ABMRegimeEnv (0.3479)

### Statistical Significance

When comparing agents, non-overlapping 95% confidence intervals indicate
statistically significant differences at the α=0.05 level.

---

*Report generated from evaluation results with confidence intervals.*
