# Training and Comparison Scripts

This directory contains scripts for training RL agents and comparing all agents across environments.

## Scripts Overview

### 1. `train_all_rl_agents.py`
Trains all RL agents (PPO, DeepPPO, LSTM-PPO) on all 12 environments and saves models.

**Usage:**
```bash
# Train all agents (skips if model exists)
python scripts/train_all_rl_agents.py

# Retrain even if model exists
python scripts/train_all_rl_agents.py --no-skip

# Custom evaluation episodes after training
python scripts/train_all_rl_agents.py --eval-episodes 200
```

**Output:**
- Models saved to: `models/{EnvName}/{AgentName}/`
- Training metadata: `models/{EnvName}/{AgentName}/metadata.json`
- Initial evaluation results: `results/{EnvName}/{AgentName}/`

### 2. `compare_all_agents.py`
Compares all agents (RL + heuristic) across all environments.

**Usage:**
```bash
# Compare all agents (skips if results exist)
python scripts/compare_all_agents.py

# Re-run even if results exist
python scripts/compare_all_agents.py --no-skip

# Only compare RL agents
python scripts/compare_all_agents.py --rl-only

# Only compare heuristic agents
python scripts/compare_all_agents.py --heuristic-only

# Custom evaluation episodes
python scripts/compare_all_agents.py --eval-episodes 2000
```

**Output:**
- Results saved to: `results/{EnvName}/{AgentName}/`
- Metrics: `results/{EnvName}/{AgentName}/metrics.json`
- PnL data: `results/{EnvName}/{AgentName}/pnls.npy`
- Plots: `results/{EnvName}/{AgentName}/pnl_distribution.png`

### 3. `aggregate_results.py`
Aggregates all results into comparison tables and reports.

**Usage:**
```bash
python scripts/aggregate_results.py

# Custom directories
python scripts/aggregate_results.py --results-dir results --output-dir results
```

**Output:**
- CSV table: `results/comparison_table.csv`
- JSON summary: `results/comparison_summary.json`
- Markdown report: `results/comparison_report.md`

### 4. `run_full_pipeline.py`
Main entry point that can run the full pipeline or individual phases.

**Usage:**
```bash
# Full pipeline: train -> compare -> aggregate
python scripts/run_full_pipeline.py --full

# Train only
python scripts/run_full_pipeline.py --train

# Compare only (assumes models exist)
python scripts/run_full_pipeline.py --compare

# Aggregate only
python scripts/run_full_pipeline.py --aggregate

# Custom options
python scripts/run_full_pipeline.py --full \
    --train-eval-episodes 100 \
    --compare-eval-episodes 1000
```

## Workflow

### Recommended Workflow

1. **Train RL Agents** (takes longest time):
   ```bash
   python scripts/train_all_rl_agents.py
   ```
   This will train 3 RL agents × 12 environments = 36 training runs.
   Each training can take 10-30 minutes depending on `total_timesteps` in config.

2. **Compare All Agents**:
   ```bash
   python scripts/compare_all_agents.py
   ```
   This evaluates all agents (RL + heuristic) on all environments.
   Much faster than training since it just loads models and evaluates.

3. **Aggregate Results**:
   ```bash
   python scripts/aggregate_results.py
   ```
   Creates comparison tables and reports.

### Quick Test Workflow

For quick testing with fewer episodes:
```bash
# Quick training (100 eval episodes)
python scripts/train_all_rl_agents.py --eval-episodes 100

# Quick comparison (100 eval episodes)
python scripts/compare_all_agents.py --eval-episodes 100

# Aggregate
python scripts/aggregate_results.py
```

## Output Structure

```
models/
  {EnvName}/
    {AgentName}/
      model.zip          # Trained model
      metadata.json       # Training config and metadata

results/
  {EnvName}/
    {AgentName}/
      metrics.json       # Performance metrics
      pnls.npy          # PnL array
      inventory.npy     # Inventory array
      pnl_distribution.png

results/
  comparison_table.csv
  comparison_summary.json
  comparison_report.md
```

## Notes

- **Training Time**: Each RL agent training can take 10-30 minutes depending on `total_timesteps` in configs.
- **Model Size**: Saved models are typically 1-10 MB each.
- **Results Size**: Each experiment result is ~100 KB (metrics + arrays).
- **Skip Logic**: Scripts skip existing models/results by default. Use `--no-skip` to force re-run.
- **Error Handling**: Scripts continue on individual failures and report summary at the end.

## Troubleshooting

### Import Errors
Make sure you're running from the project root:
```bash
cd /path/to/market_making_rl
python scripts/train_all_rl_agents.py
```

### Model Not Found
If comparison script says "Model not found", make sure training completed successfully. Check `models/` directory.

### Memory Issues
If you run out of memory during training:
- Reduce `total_timesteps` in agent configs
- Train one agent at a time
- Use smaller batch sizes

### Library Issues
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```
