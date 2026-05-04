# Environment Evaluation Report

## Executive Summary

This document evaluates all 12 market-making environments for:
1. **Mathematical correctness** of stochastic process implementations
2. **Parameter comparability** across environments
3. **Consistency** in default values
4. **Issues and bugs** requiring fixes

---

## 1. Environment Inventory

### ABM (Arithmetic Brownian Motion) Environments
- ✅ `ABMVanillaEnv` - Basic ABM: dS = μ·dt + σ·dW
- ✅ `ABMJumpEnv` - ABM with jumps
- ✅ `ABMRegimeEnv` - ABM with regime-switching volatility
- ✅ `ABMJumpRegimeEnv` - ABM with jumps + regimes

### GBM (Geometric Brownian Motion) Environments
- ✅ `GBMVanillaEnv` - Basic GBM: dS = μ·S·dt + σ·S·dW
- ✅ `GBMJumpEnv` - GBM with multiplicative (log-normal) jumps
- ✅ `GBMRegimeEnv` - GBM with regime-switching volatility
- ✅ `GBMJumpRegimeEnv` - GBM with regimes + multiplicative jumps

### OU (Ornstein-Uhlenbeck) Environments
- ✅ `OUVanillaEnv` - Basic OU: dS = κ·(μ - S)·dt + σ·dW
- ✅ `OUJumpEnv` - OU with jumps
- ✅ `OURegimeEnv` - OU with regime-switching volatility
- ✅ `OUJumpRegimeEnv` - OU with jumps + regimes

---

## 2. Bug History and Fixes

### ✅ FIXED: GBM Jump Environments - Jump Implementation

**Files:** `gbm_jump.py`, `gbm_jump_regime.py`

**Original Problem (RESOLVED):**
GBM environments previously added jumps **additively**, which was mathematically incorrect for geometric processes.

**Original (WRONG) code:**
```python
gbm_step = self.S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
if self.rng.uniform() < self.jump_intensity * dt:
    J = self._draw_jump()
else:
    J = 0.0
self.S = gbm_step + J  # ❌ WRONG: Additive jump on GBM
```

**Fixed Implementation:**
Jumps are now **multiplicative** (log-normal jumps), which is correct for geometric processes:
```python
# Current (CORRECT) code in gbm_jump.py
gbm_diffusion = (mu - 0.5 * sigma**2) * dt + sigma * dW
if self.rng.uniform() < self.jump_intensity * dt:
    J_log = self._draw_jump()  # Log-jump size ~ N(jump_mean, jump_std²)
else:
    J_log = 0.0
self.S = self.S * np.exp(gbm_diffusion + J_log)  # ✅ Multiplicative
```

**Fix Status:** ✅ **FIXED**
- Ensures price remains positive (S > 0 always)
- Mathematically consistent with geometric Brownian motion
- Follows Merton jump-diffusion model with multiplicative (log-normal) jumps

**Updated Parameters:**
- `jump_std` default changed from 1.0 to 0.05 (appropriate for log-jump size in percentage scale)
- Jumps are now properly scaled for multiplicative processes

---

## 3. Parameter Comparability Analysis

### 3.1 Volatility Parameters (σ)

| Environment Type | Default σ | Scale | Notes |
|-----------------|-----------|-------|-------|
| ABM Vanilla | 2.0 | Absolute | dS = σ·dW, so σ has units of price |
| ABM Regime Low | 1.0 | Absolute | |
| ABM Regime High | 4.0 | Absolute | 4x multiplier |
| GBM Vanilla | 0.02 | Percentage | dS = σ·S·dW, so σ is dimensionless |
| GBM Regime Low | 0.01 | Percentage | |
| GBM Regime High | 0.05 | Percentage | 5x multiplier |
| OU Vanilla | 2.0 | Absolute | Same as ABM |
| OU Regime Low | 1.0 | Absolute | |
| OU Regime High | 4.0 | Absolute | |

**Issue:** 
- **ABM/OU use absolute volatility (2.0)**
- **GBM uses percentage volatility (0.02)**
- These are **NOT directly comparable**

**For S0=100:**
- ABM σ=2.0 → std dev per step ≈ 2.0·√dt ≈ 0.2
- GBM σ=0.02 → std dev per step ≈ 0.02·100·√dt ≈ 0.2
- **They are approximately comparable at S0=100**, but this breaks if S0 changes

**Recommendation:** 
- Document that GBM σ is percentage-based
- Consider making GBM σ=0.02 → σ=2.0 for consistency (but this changes the model)

### 3.2 Jump Parameters

| Environment | jump_intensity | jump_mean | jump_std | Notes |
|-------------|----------------|-----------|----------|--------|
| ABM Jump | 0.1 | 0.0 | 5.0 | Absolute jump size |
| ABM JumpRegime | 0.1 | 0.0 | 5.0 | Absolute jump size |
| GBM Jump | 0.1 | 0.0 | 0.05 | Log-jump size (percentage scale) |
| GBM JumpRegime | 0.1 | 0.0 | 0.05 | Log-jump size (percentage scale) |
| OU Jump | 0.1 | 0.0 | 5.0 | Absolute jump size |
| OU JumpRegime | 0.1 | 0.0 | 5.0 | Absolute jump size |

**Status:**
1. ✅ **GBM jumps are now multiplicative** - Correctly implemented as log-normal jumps
2. ✅ **GBM jump_std=0.05** - Appropriate for log-jump size in percentage scale (fixed from 1.0)
3. **jump_mean=0.0 everywhere** - No directional bias (reasonable default)
4. **ABM/OU jump_std=5.0** - Absolute jump size, appropriate for additive processes

**Note:**
- GBM uses **log-jump sizes** (percentage scale) for multiplicative jumps
- ABM/OU use **absolute jump sizes** for additive jumps
- These scales are appropriate for their respective process types and are not directly comparable

### 3.3 Regime Parameters

| Environment | sigma_low | sigma_high | Ratio | Transition Matrix |
|-------------|-----------|------------|-------|-------------------|
| ABM Regime | 1.0 | 4.0 | 4x | [[0.95, 0.05], [0.10, 0.90]] |
| ABM JumpRegime | 1.0 | 4.0 | 4x | Same |
| GBM Regime | 0.01 | 0.05 | 5x | Same |
| GBM JumpRegime | 0.01 | 0.05 | 5x | Same |
| OU Regime | 1.0 | 4.0 | 4x | Same |
| OU JumpRegime | 1.0 | 4.0 | 4x | Same |

**Status:** ✅ **Consistent** - All use same transition matrix, similar volatility ratios

### 3.4 OU-Specific Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| kappa | 1.0 | Mean-reversion speed (all OU envs) |
| mu | 100.0 | Long-run mean (all OU envs) |

**Status:** ✅ **Consistent** - All OU environments use same defaults

**Note:** With S0=100.0 (from base), OU starts at long-run mean, which is reasonable.

---

## 4. Mathematical Correctness Check

### 4.1 ABM Environments ✅

**Formula:** dS = μ·dt + σ·dW

**Implementation:**
```python
dW = self.rng.normal(0.0, np.sqrt(self.dt))
self.S = self.S + self.mu * self.dt + self.sigma * dW
```

**Status:** ✅ **CORRECT** - Proper Euler-Maruyama discretization

### 4.2 GBM Environments ⚠️

**Formula:** dS = μ·S·dt + σ·S·dW

**Implementation:**
```python
dW = self.rng.normal(0.0, np.sqrt(self.dt))
self.S = self.S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
```

**Status:** ✅ **CORRECT** - Exact solution (no discretization error)

**BUT:** Jump implementations are **WRONG** (see Bug #1)

### 4.3 OU Environments ✅

**Formula:** dS = κ·(μ - S)·dt + σ·dW

**Implementation:**
```python
dW = self.rng.normal(0.0, np.sqrt(self.dt))
self.S = self.S + self.kappa * (self.mu - self.S) * self.dt + self.sigma * dW
```

**Status:** ✅ **CORRECT** - Proper Euler-Maruyama discretization

### 4.4 Jump Implementations

**ABM/OU Jumps:** ✅ **CORRECT**
- Additive jumps: S ← S + J, where J ~ N(jump_mean, jump_std²)
- Appropriate for arithmetic processes

**GBM Jumps:** ✅ **CORRECT**
- Implementation: S ← S·exp(diffusion + J_log) (multiplicative)
- J_log ~ N(jump_mean, jump_std²) represents log-jump size
- Properly ensures S > 0 and follows Merton jump-diffusion model

---

## 5. Parameter Comparability Summary

### ✅ Comparable Groups

1. **ABM & OU vanilla environments:**
   - Both use σ=2.0 (absolute)
   - Both use μ=0.0 (no drift)
   - **Directly comparable** for volatility effects

2. **All regime environments:**
   - Same transition matrix
   - Similar volatility ratios (4x or 5x)
   - **Comparable** for regime-switching effects

3. **All jump environments (except GBM):**
   - Same jump_intensity=0.1
   - Same jump_std=5.0
   - **Comparable** for jump effects

### ⚠️ Not Directly Comparable

1. **GBM vs ABM/OU:**
   - Different volatility scales (percentage vs absolute)
   - Only comparable at S0≈100
   - **Document this limitation**

2. **GBM jumps vs others:**
   - ✅ Fixed: Now correctly multiplicative (log-normal)
   - Different scales: GBM uses log-jump size (percentage, std=0.05), ABM/OU use absolute jump size (std=5.0)
   - These are appropriate for their process types but not directly comparable in magnitude

---

## 6. Recommendations

### ✅ Completed Fixes

1. ✅ **Fixed GBM jump implementations** (gbm_jump.py, gbm_jump_regime.py)
   - Changed from additive to multiplicative jumps
   - Now using log-jump sizes (J_log ~ N(jump_mean, jump_std²))
   - Updated default jump_std from 1.0 to 0.05 (appropriate for log-jump size)

### Priority 1: Documentation

1. **Document parameter scales:**
   - ✅ GBM σ is percentage-based (0.02 = 2%) - documented in code
   - ✅ ABM/OU σ is absolute (2.0 = $2 per unit time) - documented in code
   - ✅ GBM jumps use log-jump size (percentage scale, std=0.05)
   - ✅ ABM/OU jumps use absolute jump size (std=5.0)
   - Clarify when environments are comparable (only at S0≈100 for volatility)

2. **Add parameter validation:**
   - Ensure S0 > 0 for GBM
   - Ensure jump_std > 0
   - Validate transition matrix probabilities sum to 1

### Priority 2: Enhancements

1. **Add regime information to observations:**
   - Currently regime is hidden from agent
   - Could add regime indicator to observation space

2. **Consider making GBM σ absolute:**
   - Would improve comparability with ABM/OU
   - But breaks standard GBM convention (percentage volatility)

3. **Add parameter presets:**
   - Create "comparable" parameter sets for fair agent comparison
   - Document which environments should be compared together

---

## 7. Testing Recommendations

1. **Unit tests for each environment:**
   - Verify price never goes negative (GBM)
   - Verify regime transitions follow Markov chain
   - Verify jump frequencies match jump_intensity
   - Verify volatility matches expected values

2. **Statistical tests:**
   - Run long simulations, verify:
     - ABM: E[S_T] ≈ S0 + μ·T, Var(S_T) ≈ σ²·T
     - GBM: E[log(S_T/S0)] ≈ (μ - 0.5σ²)·T, Var(log(S_T/S0)) ≈ σ²·T
     - OU: E[S_T] → μ as T→∞, variance → σ²/(2κ)

3. **Comparison tests:**
   - Run same agent on ABM vs OU (should perform similarly with same σ)
   - Verify regime environments show regime-dependent behavior

---

## 8. Summary Table

| Environment | Math Correct? | Params Comparable? | Issues |
|-------------|---------------|-------------------|--------|
| ABMVanillaEnv | ✅ | ✅ | None |
| ABMJumpEnv | ✅ | ✅ | None |
| ABMRegimeEnv | ✅ | ✅ | None |
| ABMJumpRegimeEnv | ✅ | ✅ | None |
| GBMVanillaEnv | ✅ | ⚠️ | Different σ scale (percentage vs absolute) |
| GBMJumpEnv | ✅ | ⚠️ | Different σ/jump scales (percentage vs absolute) |
| GBMRegimeEnv | ✅ | ⚠️ | Different σ scale (percentage vs absolute) |
| GBMJumpRegimeEnv | ✅ | ⚠️ | Different σ/jump scales (percentage vs absolute) |
| OUVanillaEnv | ✅ | ✅ | None |
| OUJumpEnv | ✅ | ✅ | None |
| OURegimeEnv | ✅ | ✅ | None |
| OUJumpRegimeEnv | ✅ | ✅ | None |

**Overall Status:**
- ✅ **12/12 environments are mathematically correct**
- ✅ **All critical bugs have been fixed** (GBM jumps now multiplicative)
- ⚠️ **Parameter comparability:** GBM uses percentage scales while ABM/OU use absolute scales (only comparable at S0≈100)

---

## Next Steps

1. ✅ **Completed:** Fixed GBM jump bugs (multiplicative jumps implemented)
2. ✅ **Completed:** Parameter documentation in code (volatility and jump scales)
3. **Short-term:** Add unit tests for mathematical correctness
4. **Medium-term:** Add parameter validation (S0 > 0, jump_std > 0, etc.)
5. **Long-term:** Consider parameter standardization for better comparability (optional)

---

## Update History

- **2026-01-10:** Updated document to reflect fixes to GBM jump implementations
  - GBM jumps now correctly multiplicative (log-normal)
  - Updated jump_std defaults from 1.0 to 0.05 for GBM environments
  - All environments now mathematically correct
