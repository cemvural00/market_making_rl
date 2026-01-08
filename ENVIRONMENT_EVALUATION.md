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
- ❌ `GBMJumpEnv` - **BUG: Additive jumps (should be multiplicative)**
- ✅ `GBMRegimeEnv` - GBM with regime-switching volatility
- ❌ `GBMJumpRegimeEnv` - **BUG: Additive jumps (should be multiplicative)**

### OU (Ornstein-Uhlenbeck) Environments
- ✅ `OUVanillaEnv` - Basic OU: dS = κ·(μ - S)·dt + σ·dW
- ✅ `OUJumpEnv` - OU with jumps
- ✅ `OURegimeEnv` - OU with regime-switching volatility
- ✅ `OUJumpRegimeEnv` - OU with jumps + regimes

---

## 2. Critical Bugs Found

### 🚨 BUG #1: GBM Jump Environments - Incorrect Jump Implementation

**Files:** `gbm_jump.py`, `gbm_jump_regime.py`

**Problem:**
GBM environments add jumps **additively**, which is mathematically incorrect for geometric processes.

**Current (WRONG) code:**
```python
# gbm_jump.py line 43-51
gbm_step = self.S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
if self.rng.uniform() < self.jump_intensity * dt:
    J = self._draw_jump()
else:
    J = 0.0
self.S = gbm_step + J  # ❌ WRONG: Additive jump on GBM
```

**Correct implementation:**
For GBM, jumps should be **multiplicative** (log-normal jumps):
```python
gbm_step = self.S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
if self.rng.uniform() < self.jump_intensity * dt:
    J = self._draw_jump()  # This should be log-jump size
    self.S = gbm_step * np.exp(J)  # ✅ Multiplicative
else:
    self.S = gbm_step
```

**Impact:** 
- Price can go negative (impossible for GBM)
- Jump magnitudes are inconsistent with ABM/OU
- Breaks financial modeling assumptions

**Fix Required:** ✅ **HIGH PRIORITY**

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
| GBM Jump | 0.1 | 0.0 | 1.0 | **Inconsistent! Should be log-jump** |
| GBM JumpRegime | 0.1 | 0.0 | 1.0 | **Inconsistent! Should be log-jump** |
| OU Jump | 0.1 | 0.0 | 5.0 | Absolute jump size |
| OU JumpRegime | 0.1 | 0.0 | 5.0 | Absolute jump size |

**Issues:**
1. **GBM jump_std=1.0 vs others=5.0** - Inconsistent magnitude
2. **GBM jumps are additive** - Should be multiplicative (log-normal)
3. **jump_mean=0.0 everywhere** - No directional bias (reasonable default)

**Recommendation:**
- Fix GBM jump implementation (multiplicative)
- Standardize jump_std to 5.0 for all (or document why GBM is different)
- For GBM, consider using log-jump size: J_log ~ N(0, σ_jump²), then S ← S·exp(J_log)

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

**GBM Jumps:** ❌ **WRONG**
- Currently: S ← S·exp(...) + J (additive)
- Should be: S ← S·exp(...)·exp(J) = S·exp(... + J) (multiplicative)

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
   - Currently broken (additive instead of multiplicative)
   - jump_std=1.0 vs 5.0
   - **Fix required before comparison**

---

## 6. Recommendations

### Priority 1: Critical Fixes

1. **Fix GBM jump implementations** (gbm_jump.py, gbm_jump_regime.py)
   - Change from additive to multiplicative jumps
   - Consider using log-jump sizes for consistency

2. **Standardize jump parameters**
   - Either make GBM jump_std=5.0 (after fixing to multiplicative)
   - Or document why GBM uses different scale

### Priority 2: Documentation

1. **Document parameter scales:**
   - GBM σ is percentage-based (0.02 = 2%)
   - ABM/OU σ is absolute (2.0 = $2 per unit time)
   - Clarify when environments are comparable

2. **Add parameter validation:**
   - Ensure S0 > 0 for GBM
   - Ensure jump_std > 0
   - Validate transition matrix probabilities sum to 1

### Priority 3: Enhancements

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
| GBMVanillaEnv | ✅ | ⚠️ | Different σ scale |
| GBMJumpEnv | ❌ | ❌ | **Additive jumps (BUG)** |
| GBMRegimeEnv | ✅ | ⚠️ | Different σ scale |
| GBMJumpRegimeEnv | ❌ | ❌ | **Additive jumps (BUG)** |
| OUVanillaEnv | ✅ | ✅ | None |
| OUJumpEnv | ✅ | ✅ | None |
| OURegimeEnv | ✅ | ✅ | None |
| OUJumpRegimeEnv | ✅ | ✅ | None |

**Overall Status:**
- ✅ 10/12 environments are mathematically correct
- ❌ 2/12 environments have critical bugs (GBM jumps)
- ⚠️ Parameter comparability needs documentation

---

## Next Steps

1. **Immediate:** Fix GBM jump bugs
2. **Short-term:** Add parameter documentation
3. **Medium-term:** Add unit tests
4. **Long-term:** Consider parameter standardization
