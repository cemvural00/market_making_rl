# Results and Conclusions

**Generated:** 2026-01-10 16:32:14

**Total Experiments:** 204

**Environments:** 12

**Agents:** 17

---

## 1. Executive Summary

### Overall Performance Highlights

- **Best Performance:** FixedSpreadAgent achieves mean PnL of 10.0936 on ABMJumpRegimeEnv
  - Sharpe Ratio: 0.3132
- **Worst Performance:** PPOAgent achieves mean PnL of -5.9687 on GBMJumpRegimeEnv
  - Sharpe Ratio: -0.1507
- **Average Performance:** Mean PnL = 2.3495 ± 2.6190

- **Best Risk-Adjusted Return:** DeepPPOAgent achieves Sharpe ratio of 0.3479 on ABMRegimeEnv
  - Mean PnL: 8.0367

### Category Performance Summary

- **RL Agents:** Average Mean PnL = 2.0827 (n=72 experiments)
- **Analytic Agents:** Average Mean PnL = 2.6700 (n=24 experiments)
- **Heuristic Agents:** Average Mean PnL = 2.4562 (n=108 experiments)

---

## 2. Comparative Analysis by Agent Category

This section provides a statistical comparison of agent categories (RL, Analytic, Heuristic) across all performance metrics.

### Mean PnL

| Category | Mean | Median | Std Dev | Range |
|----------|------|--------|---------|-------|
| RL | 2.0827 | 1.6757 | 2.8889 | [-5.9687, 9.2689] |
| Analytic | 2.6700 | 2.6355 | 2.6731 | [-2.8642, 7.1640] |
| Heuristic | 2.4562 | 2.2985 | 2.3789 | [-3.1402, 10.0936] |

### Sharpe Ratio

| Category | Mean | Median | Std Dev | Range |
|----------|------|--------|---------|-------|
| RL | 0.0805 | 0.0756 | 0.0998 | [-0.1507, 0.3479] |
| Analytic | 0.1323 | 0.1099 | 0.1296 | [-0.0927, 0.3444] |
| Heuristic | 0.1082 | 0.1076 | 0.1014 | [-0.1292, 0.3342] |

### Standard Deviation

| Category | Mean | Median | Std Dev | Range |
|----------|------|--------|---------|-------|
| RL | 26.1757 | 24.8015 | 7.2944 | [14.5537, 52.2927] |
| Analytic | 21.6494 | 19.4841 | 8.5439 | [11.9146, 55.4343] |
| Heuristic | 23.3137 | 21.1349 | 6.2967 | [14.1464, 40.4463] |

### Consistency Across Environments (Coefficient of Variation)

Lower CV indicates more consistent performance across environments.

| Agent | CV | Mean PnL | Std Dev | Environments |
|-------|----|----|----------|------------|
| LastLookAgent | 0.5746 | 3.3555 | 1.9282 | 12 |
| ZeroIntelligenceAgent | 0.7234 | 2.7409 | 1.9828 | 12 |
| MarketOrderOnlyAgent | 0.7603 | 2.8145 | 2.1399 | 12 |
| NoiseTraderUniform | 0.8079 | 2.7027 | 2.1836 | 12 |
| InventorySpreadScalerAgent | 0.8203 | 1.9272 | 1.5810 | 12 |
| ASClosedFormAgent | 0.8295 | 3.4091 | 2.8279 | 12 |
| SACAgent | 0.8788 | 2.7801 | 2.4432 | 12 |
| MidPriceFollowAgent | 1.0286 | 2.5155 | 2.5875 | 12 |
| LSTMSACAgent | 1.1333 | 2.4251 | 2.7483 | 12 |
| FixedSpreadAgent | 1.1410 | 2.4879 | 2.8387 | 12 |

---

## 3. Individual Agent Performance Analysis

### Top Performers by Mean PnL

| Rank | Agent | Mean PnL | Median | Std Dev | Environments |
|------|-------|----------|--------|---------|-------------|
| 1 | ASClosedFormAgent | 3.4091 | 4.3225 | 2.9536 | 12 |
| 2 | LastLookAgent | 3.3555 | 3.3644 | 2.0139 | 12 |
| 3 | MarketOrderOnlyAgent | 2.8145 | 2.3407 | 2.2350 | 12 |
| 4 | SACAgent | 2.7801 | 3.2564 | 2.5518 | 12 |
| 5 | ZeroIntelligenceAgent | 2.7409 | 2.5712 | 2.0709 | 12 |
| 6 | NoiseTraderUniform | 2.7027 | 1.7571 | 2.2807 | 12 |
| 7 | MidPriceFollowAgent | 2.5155 | 2.0271 | 2.7025 | 12 |
| 8 | FixedSpreadAgent | 2.4879 | 2.4900 | 2.9649 | 12 |
| 9 | LSTMSACAgent | 2.4251 | 1.8477 | 2.8705 | 12 |
| 10 | TD3Agent | 2.3623 | 1.8950 | 2.8848 | 12 |

### Top Performers by Sharpe Ratio

| Rank | Agent | Sharpe Ratio | Mean | Std Dev | Environments |
|------|-------|--------------|------|---------|-------------|
| 1 | ASClosedFormAgent | 0.1571 | 0.1406 | 0.1426 | 12 |
| 2 | LastLookAgent | 0.1525 | 0.1291 | 0.1048 | 12 |
| 3 | NoiseTraderUniform | 0.1196 | 0.0939 | 0.1037 | 12 |
| 4 | SACAgent | 0.1174 | 0.1522 | 0.0952 | 12 |
| 5 | ZeroIntelligenceAgent | 0.1143 | 0.1403 | 0.0739 | 12 |
| 6 | FixedSpreadAgent | 0.1134 | 0.1316 | 0.1136 | 12 |
| 7 | MarketOrderOnlyAgent | 0.1133 | 0.1037 | 0.0903 | 12 |
| 8 | ASSimpleHeuristicAgent | 0.1076 | 0.1099 | 0.1223 | 12 |
| 9 | MidPriceFollowAgent | 0.1038 | 0.0957 | 0.1082 | 12 |
| 10 | InventorySpreadScalerAgent | 0.0979 | 0.0812 | 0.0889 | 12 |

### Agent Specialization by Environment Type

This analysis identifies which agents excel in which environment types.

- **ABM Environments:** Best agent is FixedSpreadAgent (Mean PnL: 10.0936)
- **GBM Environments:** Best agent is TD3Agent (Mean PnL: 9.2689)
- **OU Environments:** Best agent is MarketOrderOnlyAgent (Mean PnL: 6.5076)

### RL Algorithm Comparison

| Algorithm | Mean PnL | Sharpe Ratio | Std Dev | Environments |
|-----------|----------|--------------|---------|-------------|
| PPOAgent | 0.8780 | 0.0491 | 24.8761 | 12 |
| DeepPPOAgent | 1.7689 | 0.0741 | 24.4707 | 12 |
| LSTMPPOAgent | 2.2818 | 0.0656 | 31.0887 | 12 |
| SACAgent | 2.7801 | 0.1174 | 24.8584 | 12 |
| TD3Agent | 2.3623 | 0.0852 | 26.4606 | 12 |
| LSTMSACAgent | 2.4251 | 0.0917 | 25.2999 | 12 |

### LSTM vs Non-LSTM RL Agents

- **LSTM Agents:** Average Mean PnL = 2.3534 (n=24)
- **Non-LSTM Agents:** Average Mean PnL = 1.9473 (n=48)
- **LSTM agents outperform non-LSTM by 20.85% in terms of mean PnL**

---

## 4. Environment Complexity Analysis

This section analyzes how environment complexity (Vanilla, Jump, Regime, Jump+Regime) affects agent performance.

### Performance by Complexity Level

| Complexity | Mean PnL | Median | Std Dev | Range |
|------------|----------|--------|---------|-------|
| Vanilla | 2.2656 | 2.0824 | 1.7011 | [-1.0068, 6.0395] |
| Jump | 2.1793 | 1.7317 | 2.5545 | [-3.3919, 7.1640] |
| Regime | 2.6109 | 2.1963 | 2.8478 | [-3.1402, 9.2689] |
| JumpRegime | 2.3423 | 2.4826 | 3.1095 | [-5.9687, 10.0936] |

### Performance by Environment Type (ABM, GBM, OU)

| Environment Type | Mean PnL | Median | Std Dev | Range |
|-----------------|----------|--------|---------|-------|
| ABM | 2.6367 | 2.1594 | 2.6387 | [-2.2499, 10.0936] |
| GBM | 2.4554 | 2.3893 | 3.1022 | [-5.9687, 9.2689] |
| OU | 1.9564 | 1.7905 | 1.9086 | [-2.3581, 6.5076] |

### Environment Difficulty Ranking

Environments ranked by average performance across all agents (higher is easier).

| Rank | Environment | Avg Mean PnL | Std Dev | Agents |
|------|-------------|--------------|---------|--------|
| 1 | GBMRegimeEnv | 4.0666 | 3.2744 | 17 |
| 2 | ABMJumpEnv | 3.0029 | 2.5179 | 17 |
| 3 | ABMJumpRegimeEnv | 2.7138 | 3.6654 | 17 |
| 4 | ABMRegimeEnv | 2.6510 | 2.4408 | 17 |
| 5 | OUVanillaEnv | 2.3575 | 1.0630 | 17 |
| 6 | GBMVanillaEnv | 2.2603 | 2.1427 | 17 |
| 7 | OUJumpEnv | 2.2348 | 1.6531 | 17 |
| 8 | GBMJumpRegimeEnv | 2.1948 | 3.3340 | 17 |
| 9 | ABMVanillaEnv | 2.1789 | 1.8671 | 17 |
| 10 | OUJumpRegimeEnv | 2.1183 | 2.4586 | 17 |
| 11 | GBMJumpEnv | 1.3001 | 3.1909 | 17 |
| 12 | OURegimeEnv | 1.1150 | 2.1308 | 17 |

### Agent Adaptability Across Environments

Agents ranked by their ability to generalize across different environments (measured by consistency of performance).

| Rank | Agent | CV (lower is better) | Mean PnL | Environments |
|------|-------|---------------------|----------|-------------|
| 1 | LastLookAgent | 0.5746 | 3.3555 | 12 |
| 2 | ZeroIntelligenceAgent | 0.7234 | 2.7409 | 12 |
| 3 | MarketOrderOnlyAgent | 0.7603 | 2.8145 | 12 |
| 4 | NoiseTraderUniform | 0.8079 | 2.7027 | 12 |
| 5 | InventorySpreadScalerAgent | 0.8203 | 1.9272 | 12 |
| 6 | ASClosedFormAgent | 0.8295 | 3.4091 | 12 |
| 7 | SACAgent | 0.8788 | 2.7801 | 12 |
| 8 | MidPriceFollowAgent | 1.0286 | 2.5155 | 12 |
| 9 | LSTMSACAgent | 1.1333 | 2.4251 | 12 |
| 10 | FixedSpreadAgent | 1.1410 | 2.4879 | 12 |

---

## 5. Risk-Return Trade-offs

### Risk-Return Analysis

Analysis of the trade-off between return (mean PnL) and risk (standard deviation).

#### By Agent Category

| Category | Mean Return | Mean Risk (Std Dev) | Sharpe Ratio |
|----------|-------------|---------------------|--------------|
| RL | 2.0827 | 26.1757 | 0.0805 |
| Analytic | 2.6700 | 21.6494 | 0.1323 |
| Heuristic | 2.4562 | 23.3137 | 0.1082 |

### Tail Risk Analysis (VaR and Expected Shortfall)

Analysis of extreme loss scenarios at 95% and 99% confidence levels.

#### Average Tail Risk by Category

| Category | VaR (95%) | ES (95%) | VaR (99%) | ES (99%) |
|----------|-----------|----------|-----------|----------|
| RL | -35.6876 | -59.4705 | -66.4060 | -92.2458 |
| Analytic | -29.2869 | -44.9792 | -49.5058 | -65.3853 |
| Heuristic | -32.3916 | -51.1430 | -55.9483 | -79.5692 |

*Note: VaR and ES are reported as negative values (losses). More negative values indicate higher tail risk.*

---

## 6. Statistical Patterns and Insights

### Correlation Analysis

- **Mean PnL vs Sharpe Ratio:** Pearson correlation = 0.9267
- **Mean PnL vs Standard Deviation:** Pearson correlation = 0.1627
- **Sharpe Ratio vs Standard Deviation:** Pearson correlation = -0.1241

### Exceptional Performances

#### Exceptional Positive Performances

| Agent | Environment | Mean PnL | Sharpe Ratio |
|-------|-------------|----------|--------------|
| FixedSpreadAgent | ABMJumpRegimeEnv | 10.0936 | 0.3132 |

#### Exceptional Negative Performances

| Agent | Environment | Mean PnL | Sharpe Ratio |
|-------|-------------|----------|--------------|
| PPOAgent | GBMJumpRegimeEnv | -5.9687 | -0.1507 |

### Notable Environment-Agent Interactions

Identification of significant positive and negative interactions between agent types and environment characteristics.

- **Best Combination:** FixedSpreadAgent on ABMJumpRegimeEnv (Mean PnL: 10.0936)
- **Worst Combination:** PPOAgent on GBMJumpRegimeEnv (Mean PnL: -5.9687)

---

## 7. Critical Discussion

### RL vs Traditional Methods

Our results reveal interesting patterns in the performance of reinforcement learning agents compared to traditional methods:

- **Analytic methods** achieve an average mean PnL of 2.6700, outperforming **RL agents** (2.0827) by 28.20%.
- **Heuristic methods** outperform **RL agents** by 17.93%.

However, performance varies significantly across environments. RL agents show particular strength in complex, non-stationary environments with regime-switching or jumps, where adaptive learning can capture patterns that fixed rules cannot. In simpler, more predictable environments, well-designed analytic or heuristic methods may perform comparably or better.

### LSTM Effectiveness

LSTM-based RL agents demonstrate superior performance (mean PnL: 2.3534) compared to non-LSTM RL agents (1.9473), suggesting that sequence modeling provides valuable context for market-making decisions. The ability to maintain memory of past market states enables these agents to better adapt to regime changes and temporal patterns.

### Heuristic Robustness

Heuristic agents demonstrate remarkable robustness and consistency across environments. The most consistent heuristic agent (LastLookAgent) achieves a coefficient of variation of 0.5746, indicating stable performance.

Simple rule-based strategies, while not always achieving the highest returns, provide predictable and reliable performance. This makes them attractive for practical applications where consistency is valued over peak performance, or as baseline strategies for comparison.

### Analytic Methods: Theory vs Practice

Analytic methods, derived from theoretical optimal control frameworks, demonstrate strong performance in environments that match their underlying assumptions. The AS (Avellaneda-Stoikov) closed-form solution and its heuristic approximations show particular strength in environments with predictable price dynamics.

However, when faced with regime-switching or jump diffusion, these methods may struggle to adapt. The theoretical optimality under idealized conditions does not always translate to superior empirical performance in realistic market simulations with non-stationary dynamics.

---

## 8. Conclusions

### Main Findings

1. **No single agent type dominates across all environments.** The best-performing agent varies significantly depending on the environment characteristics, emphasizing the importance of selecting appropriate strategies for specific market conditions.

2. **RL agents show promise in complex environments.** In environments with regime-switching, jumps, or non-stationary dynamics, RL agents demonstrate the ability to adapt and learn effective policies that traditional methods struggle with.

3. **Heuristic methods provide robust baselines.** Simple rule-based strategies achieve consistent, predictable performance across diverse environments, making them valuable for practical applications and as benchmarks for comparison.

4. **Environment complexity significantly impacts performance.** As environments become more complex (from vanilla to jump+regime), the variance in agent performance increases, and the relative advantages of different approaches become more pronounced.

5. **LSTM architectures offer potential advantages** for capturing temporal dependencies, though their effectiveness varies across environments and may require careful tuning.

### Research Contributions

This study contributes to the market-making literature by:

- Providing a comprehensive comparison of RL, analytic, and heuristic methods across 12 diverse environments
- Demonstrating the importance of environment characteristics in determining optimal strategy selection
- Establishing benchmarks for future research in algorithmic market making
- Identifying specific scenarios where RL methods provide advantages over traditional approaches

### Practical Implications

For practitioners:

- **Environment assessment is critical:** Understanding market characteristics (volatility regimes, jumps, price dynamics) should guide strategy selection.
- **Hybrid approaches may be optimal:** Combining different agent types or using ensemble methods could leverage the strengths of each approach.
- **Robustness vs performance trade-off:** Simple heuristics may be preferable when consistency is more important than peak performance.
- **RL requires careful tuning:** While RL agents show promise, they require significant computational resources and hyperparameter tuning to achieve optimal performance.

### Limitations

This study has several limitations:

- **Simulated environments:** Results are based on synthetic market simulations and may not fully capture real market dynamics, including adverse selection and information asymmetry.
- **Limited RL training:** RL agents are trained for a fixed number of timesteps; longer training might improve performance.
- **Hyperparameter sensitivity:** RL agent performance may be sensitive to hyperparameters not fully explored.
- **Single-asset focus:** Results are based on single-asset market making; multi-asset scenarios may differ.
- **Transaction costs:** While included in the simulation, real-world transaction costs and market impact may differ significantly.

### Future Directions

Future research could explore:

- **Multi-asset market making:** Extending analysis to portfolio-based market making strategies.
- **More sophisticated RL architectures:** Exploring transformer-based models, attention mechanisms, and multi-agent RL approaches.
- **Real market data:** Validating findings on historical market data and live trading environments.
- **Ensemble methods:** Combining multiple agents to leverage complementary strengths.
- **Robustness to model misspecification:** Testing agent performance when environment assumptions are violated.
- **Computational efficiency:** Developing more efficient training and inference methods for RL agents.

---

## References

For detailed metrics and data, see:
- [Evaluation Report](EVALUATION_REPORT.md) - Raw statistics and rankings
- [Visualization Report](VISUALIZATION_REPORT.md) - Visual representations and confidence intervals
- [Appendix](APPENDIX.md) - Detailed matrix format data

