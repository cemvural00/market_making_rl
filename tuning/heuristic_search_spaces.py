"""
Optuna search space definitions for non-RL (analytic + heuristic) agents.

Each sampler takes (trial, env_config) and returns a config dict suitable for
agent_class(config=...).  env_config is passed so that environment-specific
constants (sigma, base_delta, max_inventory) can be pinned rather than tuned.

Academic rationale
------------------
For the AS closed-form agent, σ is fixed to the environment's known value.
AS is a market-calibrated model: it *knows* the volatility process. Tuning σ
would be epistemically wrong and would also require incompatible search ranges
across env types (ABM σ≈20 vs GBM σ≈0.02). Only γ (risk aversion) and k
(intensity decay) are genuine free parameters that a practitioner calibrates.

For heuristic agents (FixedSpread, InventoryShift, etc.) all parameters are
design choices with no canonical defaults. Tuning them gives each heuristic its
best possible performance — a prerequisite for a fair cross-agent comparison.

50 trials per combo is used (vs 30 for RL) because heuristic trials have no
training cost (~1-3s each) and more trials improve TPE coverage at negligible
wall-clock cost.
"""

# ── Search space functions ────────────────────────────────────────────────────

def sample_as_closed_form(trial, env_config):
    """
    ASClosedFormAgent: tune γ and k only.

    σ is fixed to the environment's actual volatility (AS knows the market).
    base_delta and max_inventory are pinned via single-value categoricals so
    they appear in trial.params and get exported to the best-params YAML.
    """
    sigma         = env_config.get("sigma", 2.0)
    base_delta    = env_config.get("base_delta", 1.0)
    max_inventory = env_config.get("max_inventory", 20)
    return {
        "gamma":         trial.suggest_float("gamma", 0.01, 1.0, log=True),
        "k":             trial.suggest_float("k", 0.3, 5.0, log=True),
        "sigma":         trial.suggest_categorical("sigma",         [sigma]),
        "base_delta":    trial.suggest_categorical("base_delta",    [base_delta]),
        "max_inventory": trial.suggest_categorical("max_inventory", [max_inventory]),
    }


def sample_as_simple(trial, env_config):
    """
    ASSimpleHeuristicAgent: tune γ and spread bounds.

    γ controls inventory skew; the spread bounds are purely design choices
    with no theoretical default.
    """
    max_inventory = env_config.get("max_inventory", 20)
    return {
        "gamma":             trial.suggest_float("gamma", 0.01, 1.0, log=True),
        "min_spread_factor": trial.suggest_float("min_spread_factor", 0.0, 0.4),
        "max_spread_factor": trial.suggest_float("max_spread_factor", 0.5, 1.0),
        "max_inventory":     trial.suggest_categorical("max_inventory", [max_inventory]),
    }


def sample_fixed_spread(trial, env_config):
    """
    FixedSpreadAgent: tune the spread multiplier.

    The only free parameter. 1.0 is a placeholder default, not a calibrated
    value. The optimal fixed spread depends on A, k, σ of the environment.
    """
    return {
        "fixed_multiplier": trial.suggest_float("fixed_multiplier", 0.3, 3.0),
    }


def sample_inv_shift(trial, env_config):
    """
    InventoryShiftAgent: tune the skew-sensitivity parameter β.

    β controls how strongly inventory tilts the quotes. Log-uniform so small
    β values (gentle skew) get equal representation with large ones.
    """
    max_inventory = env_config.get("max_inventory", 20)
    return {
        "beta":          trial.suggest_float("beta", 0.005, 0.5, log=True),
        "max_inventory": trial.suggest_categorical("max_inventory", [max_inventory]),
    }


def sample_inv_spread_scaler(trial, env_config):
    """
    InventorySpreadScalerAgent: tune the spread-sensitivity parameter α.

    α controls how much |inventory| widens the spread. Same log-uniform
    rationale as for β in InventoryShiftAgent.
    """
    max_inventory = env_config.get("max_inventory", 20)
    return {
        "alpha":         trial.suggest_float("alpha", 0.005, 0.5, log=True),
        "max_inventory": trial.suggest_categorical("max_inventory", [max_inventory]),
    }


def sample_last_look(trial, env_config):
    """
    LastLookAgent: tune trend sensitivity.

    Controls how strongly recent price change (dS) shifts the quotes.
    Wide range captures both sluggish and hyper-reactive settings.
    """
    return {
        "trend_sensitivity": trial.suggest_float("trend_sensitivity", 0.05, 3.0),
    }


def sample_mid_price_follow(trial, env_config):
    """
    MidPriceFollowAgent: tune trend sensitivity and max skew cap.

    trend_sensitivity: strength of reaction to dS
    max_skew:          hard cap on the skew action (prevents extreme positions)
    """
    return {
        "trend_sensitivity": trial.suggest_float("trend_sensitivity", 0.05, 3.0),
        "max_skew":          trial.suggest_float("max_skew", 0.1, 1.0),
    }


# ── Registry ─────────────────────────────────────────────────────────────────

HEURISTIC_SEARCH_SPACE_REGISTRY = {
    "ASClosedFormAgent":          sample_as_closed_form,
    "ASSimpleHeuristicAgent":     sample_as_simple,
    "FixedSpreadAgent":           sample_fixed_spread,
    "InventoryShiftAgent":        sample_inv_shift,
    "InventorySpreadScalerAgent": sample_inv_spread_scaler,
    "LastLookAgent":              sample_last_look,
    "MidPriceFollowAgent":        sample_mid_price_follow,
}

# Agents deliberately excluded from tuning:
#   MarketOrderOnlyAgent   — no parameters; deterministic by construction
#   NoiseTraderNormal      — random agent; tuning would remove the noise
#   NoiseTraderUniform     — same argument
#   ZeroIntelligenceAgent  — pure Uniform[-1,1] random; no parameters


def get_heuristic_search_space_fn(agent_name):
    """Return the sampler function for the given non-RL agent class name."""
    if agent_name not in HEURISTIC_SEARCH_SPACE_REGISTRY:
        raise ValueError(
            f"No heuristic search space defined for agent: {agent_name}. "
            f"Available: {list(HEURISTIC_SEARCH_SPACE_REGISTRY.keys())}"
        )
    return HEURISTIC_SEARCH_SPACE_REGISTRY[agent_name]
