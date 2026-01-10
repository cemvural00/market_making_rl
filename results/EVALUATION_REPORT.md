# Market Making Agent Evaluation Report

**Generated:** 2026-01-10 10:16:24

**Total Experiments:** 204

**Environments:** 12

**Agents:** 17

## Agent Categories

- **RL Agents:** 6 (PPOAgent, DeepPPOAgent, LSTMPPOAgent, SACAgent, TD3Agent, LSTMSACAgent)
- **Analytic Agents:** 2 (ASClosedFormAgent, ASSimpleHeuristicAgent)
- **Heuristic Agents:** 9 (ZeroIntelligenceAgent, MidPriceFollowAgent, NoiseTraderNormal, FixedSpreadAgent, NoiseTraderUniform, InventorySpreadScalerAgent, InventoryShiftAgent, MarketOrderOnlyAgent, LastLookAgent)

## Environment Types

- **ABM:** 4 environments
- **GBM:** 4 environments
- **OU:** 4 environments

## Overall Performance Statistics

- **Best Mean PnL:** 10.0936 (FixedSpreadAgent on ABMJumpRegimeEnv)
- **Worst Mean PnL:** -5.9687 (PPOAgent on GBMJumpRegimeEnv)
- **Average Mean PnL:** 2.3495
- **Std Dev of Mean PnL:** 2.6190

- **Best Sharpe Ratio:** 0.3479 (DeepPPOAgent on ABMRegimeEnv)
- **Worst Sharpe Ratio:** -0.1507 (PPOAgent on GBMJumpRegimeEnv)
- **Average Sharpe Ratio:** 0.1013

## Best Agent per Environment

### By Mean PnL

| Environment | Best Agent | Mean PnL | Sharpe | Std Dev |
|-------------|------------|----------|--------|----------|
| ABMJumpEnv | ASClosedFormAgent | 6.6437 | 0.3404 | 19.5174 |
| ABMJumpRegimeEnv | FixedSpreadAgent | 10.0936 | 0.3132 | 32.2229 |
| ABMRegimeEnv | LSTMPPOAgent | 8.9675 | 0.3257 | 27.5305 |
| ABMVanillaEnv | ASClosedFormAgent | 5.7641 | 0.3444 | 16.7386 |
| GBMJumpEnv | ASClosedFormAgent | 7.1640 | 0.2567 | 27.9122 |
| GBMJumpRegimeEnv | MarketOrderOnlyAgent | 6.3433 | 0.1568 | 40.4463 |
| GBMRegimeEnv | TD3Agent | 9.2689 | 0.2456 | 37.7416 |
| GBMVanillaEnv | InventoryShiftAgent | 6.0395 | 0.3229 | 18.7049 |
| OUJumpEnv | NoiseTraderUniform | 4.9233 | 0.3257 | 15.1168 |
| OUJumpRegimeEnv | MarketOrderOnlyAgent | 6.5076 | 0.3142 | 20.7143 |
| OURegimeEnv | LastLookAgent | 5.1719 | 0.1969 | 26.2676 |
| OUVanillaEnv | SACAgent | 4.7365 | 0.2069 | 22.8957 |

### By Sharpe Ratio

| Environment | Best Agent | Sharpe | Mean PnL | Std Dev |
|-------------|------------|--------|----------|----------|
| ABMJumpEnv | ASClosedFormAgent | 0.3404 | 6.6437 | 19.5174 |
| ABMJumpRegimeEnv | MidPriceFollowAgent | 0.3342 | 8.4474 | 25.2749 |
| ABMRegimeEnv | DeepPPOAgent | 0.3479 | 8.0367 | 23.1020 |
| ABMVanillaEnv | ASClosedFormAgent | 0.3444 | 5.7641 | 16.7386 |
| GBMJumpEnv | LastLookAgent | 0.3298 | 6.9801 | 21.1633 |
| GBMJumpRegimeEnv | MidPriceFollowAgent | 0.2086 | 4.6899 | 22.4830 |
| GBMRegimeEnv | TD3Agent | 0.2456 | 9.2689 | 37.7416 |
| GBMVanillaEnv | ASSimpleHeuristicAgent | 0.3320 | 5.8293 | 17.5556 |
| OUJumpEnv | NoiseTraderUniform | 0.3257 | 4.9233 | 15.1168 |
| OUJumpRegimeEnv | ASClosedFormAgent | 0.3295 | 5.3669 | 16.2876 |
| OURegimeEnv | NoiseTraderUniform | 0.2088 | 4.9235 | 23.5839 |
| OUVanillaEnv | SACAgent | 0.2069 | 4.7365 | 22.8957 |

## Best Environment per Agent

### By Mean PnL

| Agent | Best Environment | Mean PnL | Sharpe | Std Dev |
|-------|------------------|----------|--------|----------|
| ASClosedFormAgent | GBMJumpEnv | 7.1640 | 0.2567 | 27.9122 |
| ASSimpleHeuristicAgent | GBMVanillaEnv | 5.8293 | 0.3320 | 17.5556 |
| DeepPPOAgent | ABMRegimeEnv | 8.0367 | 0.3479 | 23.1020 |
| FixedSpreadAgent | ABMJumpRegimeEnv | 10.0936 | 0.3132 | 32.2229 |
| InventoryShiftAgent | GBMRegimeEnv | 6.1661 | 0.1610 | 38.3061 |
| InventorySpreadScalerAgent | ABMJumpEnv | 4.6474 | 0.2497 | 18.6146 |
| LSTMPPOAgent | ABMRegimeEnv | 8.9675 | 0.3257 | 27.5305 |
| LSTMSACAgent | GBMRegimeEnv | 8.8750 | 0.2429 | 36.5448 |
| LastLookAgent | GBMJumpEnv | 6.9801 | 0.3298 | 21.1633 |
| MarketOrderOnlyAgent | OUJumpRegimeEnv | 6.5076 | 0.3142 | 20.7143 |
| MidPriceFollowAgent | ABMJumpRegimeEnv | 8.4474 | 0.3342 | 25.2749 |
| NoiseTraderNormal | ABMJumpEnv | 6.3318 | 0.2178 | 29.0668 |
| NoiseTraderUniform | ABMJumpRegimeEnv | 6.3073 | 0.2555 | 24.6851 |
| PPOAgent | GBMRegimeEnv | 4.9052 | 0.1344 | 36.4974 |
| SACAgent | GBMRegimeEnv | 6.0406 | 0.1424 | 42.4071 |
| TD3Agent | GBMRegimeEnv | 9.2689 | 0.2456 | 37.7416 |
| ZeroIntelligenceAgent | GBMRegimeEnv | 6.0010 | 0.1539 | 39.0047 |

## Performance by Agent Category

### RL Agents

- **Average Mean PnL:** 2.0827
- **Average Sharpe:** 0.0805
- **Best RL Agent:** TD3Agent (mean: 9.2689)

### Analytic Agents

- **Average Mean PnL:** 2.6700
- **Average Sharpe:** 0.1323
- **Best Analytic Agent:** ASClosedFormAgent (mean: 7.1640)

### Heuristic Agents

- **Average Mean PnL:** 2.4562
- **Average Sharpe:** 0.1082
- **Best Heuristic Agent:** FixedSpreadAgent (mean: 10.0936)

## Performance by Environment Type

### ABM Environments

- **Average Mean PnL:** 2.6367
- **Average Sharpe:** 0.1113
- **Best Agent:** FixedSpreadAgent on ABMJumpRegimeEnv (mean: 10.0936)

### GBM Environments

- **Average Mean PnL:** 2.4554
- **Average Sharpe:** 0.0880
- **Best Agent:** TD3Agent on GBMRegimeEnv (mean: 9.2689)

### OU Environments

- **Average Mean PnL:** 1.9564
- **Average Sharpe:** 0.1044
- **Best Agent:** MarketOrderOnlyAgent on OUJumpRegimeEnv (mean: 6.5076)

## Risk Analysis

### Value at Risk (VaR) and Expected Shortfall (ES)

| Agent | Avg VaR (95%) | Avg ES (95%) | Avg VaR (99%) | Avg ES (99%) |
|-------|---------------|--------------|---------------|--------------|
| ASClosedFormAgent | -34.0419 | -51.4999 | -55.6276 | -70.8373 |
| ASSimpleHeuristicAgent | -24.5318 | -38.4585 | -43.3840 | -59.9334 |
| DeepPPOAgent | -33.4900 | -57.2792 | -69.3688 | -87.9621 |
| FixedSpreadAgent | -29.7886 | -45.2333 | -49.0930 | -72.4755 |
| InventoryShiftAgent | -31.7391 | -52.5726 | -58.9758 | -78.8992 |
| InventorySpreadScalerAgent | -32.7211 | -51.2336 | -55.2717 | -74.3588 |
| LSTMPPOAgent | -40.2544 | -71.3173 | -77.9956 | -117.7392 |
| LSTMSACAgent | -37.1619 | -57.0948 | -63.3696 | -81.9222 |
| LastLookAgent | -30.0839 | -47.9371 | -49.9979 | -71.6178 |
| MarketOrderOnlyAgent | -33.6085 | -52.3246 | -57.3078 | -80.4870 |
| MidPriceFollowAgent | -31.9328 | -50.3149 | -52.4893 | -79.7726 |
| NoiseTraderNormal | -37.0473 | -56.5565 | -62.9955 | -90.9171 |
| NoiseTraderUniform | -32.2642 | -51.5677 | -57.4021 | -80.0759 |
| PPOAgent | -35.4932 | -57.9040 | -61.5810 | -89.9470 |
| SACAgent | -32.2244 | -55.6695 | -60.9745 | -86.8210 |
| TD3Agent | -35.5015 | -57.5580 | -65.1465 | -89.0836 |
| ZeroIntelligenceAgent | -32.3391 | -52.5469 | -60.0014 | -87.5190 |

## Inventory Management

### Average Inventory Levels

| Agent | Avg Inventory (across all envs) |
|-------|----------------------------------|
| ASClosedFormAgent | 1.2492 |
| ASSimpleHeuristicAgent | 0.8417 |
| DeepPPOAgent | 1.1825 |
| FixedSpreadAgent | 1.0942 |
| InventoryShiftAgent | 1.0308 |
| InventorySpreadScalerAgent | 1.1108 |
| LSTMPPOAgent | 1.5800 |
| LSTMSACAgent | 1.3075 |
| LastLookAgent | 1.1083 |
| MarketOrderOnlyAgent | 1.0758 |
| MidPriceFollowAgent | 1.1100 |
| NoiseTraderNormal | 1.1733 |
| NoiseTraderUniform | 1.1658 |
| PPOAgent | 1.3042 |
| SACAgent | 1.2733 |
| TD3Agent | 1.3917 |
| ZeroIntelligenceAgent | 1.1942 |

## Detailed Per-Environment Results

### ABMJumpEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| ASClosedFormAgent | 6.6437 | 19.5174 | 0.3404 | -19.8874 | -35.2415 | 1.1200 |
| NoiseTraderNormal | 6.3318 | 29.0668 | 0.2178 | -35.9983 | -45.1799 | 1.2400 |
| SACAgent | 6.0250 | 23.6082 | 0.2552 | -15.3194 | -54.3165 | 1.3900 |
| LSTMSACAgent | 6.0033 | 24.9140 | 0.2410 | -30.3584 | -44.5827 | 1.3600 |
| LastLookAgent | 5.3124 | 20.2355 | 0.2625 | -18.4064 | -36.3169 | 0.8400 |
| ZeroIntelligenceAgent | 5.1835 | 22.5292 | 0.2301 | -25.5895 | -35.7818 | 1.3600 |
| InventorySpreadScalerAgent | 4.6474 | 18.6146 | 0.2497 | -18.5004 | -41.7328 | 1.1200 |
| InventoryShiftAgent | 2.3011 | 20.6770 | 0.1113 | -32.2341 | -48.7882 | 1.1100 |
| MarketOrderOnlyAgent | 2.2801 | 23.4644 | 0.0972 | -29.4840 | -41.6025 | 1.0400 |
| PPOAgent | 2.2511 | 15.5503 | 0.1448 | -20.3744 | -33.0098 | 1.1100 |
| NoiseTraderUniform | 1.4236 | 19.7082 | 0.0722 | -26.6580 | -47.3453 | 1.2600 |
| MidPriceFollowAgent | 1.2797 | 22.6103 | 0.0566 | -37.9412 | -45.8840 | 1.0600 |
| LSTMPPOAgent | 1.2557 | 36.2646 | 0.0346 | -45.7264 | -110.1553 | 1.6100 |
| ASSimpleHeuristicAgent | 0.6262 | 12.9704 | 0.0483 | -19.9691 | -27.4431 | 0.8800 |
| TD3Agent | 0.0485 | 21.5302 | 0.0023 | -28.2395 | -48.6598 | 1.0200 |
| FixedSpreadAgent | -0.0379 | 21.8208 | -0.0017 | -34.3551 | -46.2816 | 1.1800 |
| DeepPPOAgent | -0.5254 | 28.5287 | -0.0184 | -42.6052 | -76.3735 | 1.8100 |

### ABMJumpRegimeEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| FixedSpreadAgent | 10.0936 | 32.2229 | 0.3132 | -34.5509 | -47.8483 | 1.3100 |
| MidPriceFollowAgent | 8.4474 | 25.2749 | 0.3342 | -26.5090 | -44.1408 | 1.1500 |
| LSTMPPOAgent | 6.8374 | 39.6309 | 0.1725 | -42.5697 | -66.8119 | 1.8700 |
| NoiseTraderUniform | 6.3073 | 24.6851 | 0.2555 | -36.3928 | -46.0643 | 1.1200 |
| ASClosedFormAgent | 4.6890 | 23.8799 | 0.1964 | -34.0162 | -43.9384 | 1.1000 |
| MarketOrderOnlyAgent | 3.8385 | 31.6307 | 0.1214 | -39.8604 | -67.2743 | 1.2900 |
| TD3Agent | 3.7983 | 30.3118 | 0.1253 | -38.5983 | -61.1597 | 1.3600 |
| ZeroIntelligenceAgent | 3.6034 | 24.9269 | 0.1446 | -36.5684 | -61.1588 | 1.1100 |
| DeepPPOAgent | 1.7900 | 37.0147 | 0.0484 | -61.0806 | -94.6675 | 1.4900 |
| LastLookAgent | 1.0010 | 21.4244 | 0.0467 | -37.9130 | -48.8579 | 1.2100 |
| SACAgent | 0.9170 | 23.2788 | 0.0394 | -36.0048 | -61.6777 | 1.1100 |
| InventoryShiftAgent | 0.9101 | 22.9627 | 0.0396 | -37.3725 | -65.0859 | 0.8800 |
| PPOAgent | -0.1324 | 31.5937 | -0.0042 | -39.0237 | -81.8505 | 1.4200 |
| InventorySpreadScalerAgent | -0.6837 | 26.2381 | -0.0261 | -50.8526 | -73.8982 | 1.0700 |
| LSTMSACAgent | -1.1683 | 26.2837 | -0.0444 | -45.3724 | -74.7876 | 1.2300 |
| ASSimpleHeuristicAgent | -1.8644 | 23.3325 | -0.0799 | -31.6611 | -59.5950 | 0.8500 |
| NoiseTraderNormal | -2.2499 | 25.2513 | -0.0891 | -42.0227 | -74.8811 | 1.2000 |

### ABMRegimeEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| LSTMPPOAgent | 8.9675 | 27.5305 | 0.3257 | -26.6715 | -39.4334 | 1.1300 |
| DeepPPOAgent | 8.0367 | 23.1020 | 0.3479 | -20.0305 | -39.6243 | 1.1600 |
| SACAgent | 3.5462 | 19.7276 | 0.1798 | -28.3702 | -33.5408 | 1.0100 |
| NoiseTraderNormal | 3.1352 | 26.5874 | 0.1179 | -33.2110 | -60.2003 | 1.1900 |
| ASSimpleHeuristicAgent | 2.5857 | 23.6851 | 0.1092 | -37.7893 | -47.2839 | 0.7200 |
| MidPriceFollowAgent | 2.5711 | 23.2223 | 0.1107 | -36.2063 | -54.2274 | 1.0800 |
| InventorySpreadScalerAgent | 2.5203 | 20.0582 | 0.1257 | -27.4886 | -36.6658 | 1.0400 |
| LSTMSACAgent | 2.3744 | 23.6452 | 0.1004 | -37.0987 | -56.6332 | 1.1800 |
| InventoryShiftAgent | 2.1963 | 33.4261 | 0.0657 | -45.0326 | -79.4839 | 1.0300 |
| LastLookAgent | 2.0555 | 32.5111 | 0.0632 | -38.0161 | -67.7876 | 1.1800 |
| ZeroIntelligenceAgent | 2.0139 | 25.9332 | 0.0777 | -34.2654 | -57.0655 | 1.1000 |
| PPOAgent | 1.9591 | 28.1397 | 0.0696 | -39.5363 | -63.4066 | 1.3800 |
| ASClosedFormAgent | 1.5586 | 26.2201 | 0.0594 | -35.2305 | -69.3946 | 1.1500 |
| NoiseTraderUniform | 1.5211 | 33.6579 | 0.0452 | -51.2619 | -81.8086 | 1.2300 |
| MarketOrderOnlyAgent | 0.4904 | 26.5721 | 0.0185 | -44.7981 | -62.2792 | 1.0400 |
| TD3Agent | -0.0469 | 27.9004 | -0.0017 | -46.7844 | -74.0762 | 1.7000 |
| FixedSpreadAgent | -0.4174 | 20.4953 | -0.0204 | -35.1970 | -47.0715 | 1.1100 |

### ABMVanillaEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| ASClosedFormAgent | 5.7641 | 16.7386 | 0.3444 | -16.7556 | -33.8704 | 1.1000 |
| FixedSpreadAgent | 4.7912 | 20.2018 | 0.2372 | -25.9930 | -38.6235 | 1.0400 |
| LastLookAgent | 4.1483 | 26.0870 | 0.1590 | -23.6774 | -52.4869 | 1.1100 |
| InventorySpreadScalerAgent | 4.0838 | 19.3205 | 0.2114 | -25.4394 | -37.0511 | 1.1300 |
| LSTMSACAgent | 3.3442 | 25.9835 | 0.1287 | -36.9413 | -53.6352 | 1.4900 |
| SACAgent | 2.9667 | 18.3251 | 0.1619 | -20.2448 | -42.0510 | 1.0800 |
| NoiseTraderUniform | 2.8789 | 21.3395 | 0.1349 | -28.7246 | -53.3895 | 1.1900 |
| PPOAgent | 2.1224 | 18.7722 | 0.1131 | -20.9042 | -33.2942 | 1.1300 |
| NoiseTraderNormal | 1.8485 | 19.9861 | 0.0925 | -32.6467 | -43.0915 | 1.1500 |
| DeepPPOAgent | 1.7645 | 19.9179 | 0.0886 | -21.9311 | -30.3556 | 0.7600 |
| TD3Agent | 1.4730 | 27.4580 | 0.0536 | -38.9704 | -53.3186 | 1.4400 |
| ZeroIntelligenceAgent | 1.3896 | 19.5479 | 0.0711 | -28.7233 | -42.3480 | 1.2900 |
| MidPriceFollowAgent | 1.2707 | 19.1437 | 0.0664 | -34.9158 | -38.6063 | 1.0500 |
| ASSimpleHeuristicAgent | 0.5265 | 13.8449 | 0.0380 | -21.2118 | -28.9393 | 0.9500 |
| LSTMPPOAgent | -0.0011 | 23.8407 | -0.0000 | -49.8604 | -65.3194 | 1.3900 |
| InventoryShiftAgent | -0.3235 | 15.8808 | -0.0204 | -23.2188 | -37.0158 | 1.1000 |
| MarketOrderOnlyAgent | -1.0068 | 24.9198 | -0.0404 | -44.3128 | -69.5238 | 1.2800 |

### GBMJumpEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| ASClosedFormAgent | 7.1640 | 27.9122 | 0.2567 | -35.1471 | -46.8655 | 1.4700 |
| LastLookAgent | 6.9801 | 21.1633 | 0.3298 | -20.3897 | -32.1081 | 1.1100 |
| SACAgent | 4.8090 | 29.4806 | 0.1631 | -30.6473 | -52.5362 | 1.5400 |
| MarketOrderOnlyAgent | 4.3059 | 21.1065 | 0.2040 | -29.4901 | -36.4879 | 1.0800 |
| LSTMSACAgent | 3.4818 | 20.8500 | 0.1670 | -33.8134 | -42.6138 | 1.3300 |
| DeepPPOAgent | 2.5158 | 18.1763 | 0.1384 | -19.6928 | -39.2748 | 1.0700 |
| MidPriceFollowAgent | 1.7038 | 19.6862 | 0.0865 | -26.7584 | -40.6260 | 1.2200 |
| LSTMPPOAgent | 1.0308 | 35.9508 | 0.0287 | -60.9810 | -93.4615 | 1.5600 |
| InventorySpreadScalerAgent | 0.9169 | 18.5050 | 0.0495 | -31.9597 | -42.9467 | 1.1200 |
| TD3Agent | 0.3902 | 28.6369 | 0.0136 | -39.8022 | -81.6335 | 1.5500 |
| NoiseTraderUniform | -0.1473 | 23.2656 | -0.0063 | -32.1318 | -55.9276 | 1.1900 |
| ASSimpleHeuristicAgent | -0.4709 | 18.7498 | -0.0251 | -30.4345 | -47.8676 | 0.9900 |
| ZeroIntelligenceAgent | -1.1993 | 26.3825 | -0.0455 | -46.3721 | -56.0959 | 1.3400 |
| FixedSpreadAgent | -1.2265 | 19.1369 | -0.0641 | -27.8761 | -49.4355 | 1.1200 |
| NoiseTraderNormal | -1.8804 | 20.0534 | -0.0938 | -39.4207 | -45.7073 | 1.2400 |
| InventoryShiftAgent | -2.8805 | 22.2898 | -0.1292 | -38.1434 | -65.6506 | 0.9800 |
| PPOAgent | -3.3919 | 25.2471 | -0.1343 | -48.4455 | -70.5259 | 1.3800 |

### GBMJumpRegimeEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| MarketOrderOnlyAgent | 6.3433 | 40.4463 | 0.1568 | -42.2296 | -73.9978 | 0.9600 |
| NoiseTraderUniform | 6.1369 | 34.1222 | 0.1799 | -39.5024 | -56.4335 | 1.2200 |
| DeepPPOAgent | 6.0599 | 33.3550 | 0.1817 | -35.0471 | -61.8121 | 1.1500 |
| ZeroIntelligenceAgent | 5.1123 | 32.1608 | 0.1590 | -30.4181 | -78.4985 | 1.1800 |
| MidPriceFollowAgent | 4.6899 | 22.4830 | 0.2086 | -37.0657 | -51.6290 | 1.0600 |
| ASSimpleHeuristicAgent | 3.8809 | 20.6535 | 0.1879 | -23.9351 | -34.2010 | 0.6700 |
| FixedSpreadAgent | 3.1551 | 35.9127 | 0.0879 | -49.7017 | -84.5665 | 0.9900 |
| InventoryShiftAgent | 2.9341 | 30.6285 | 0.0958 | -39.8755 | -68.1100 | 0.9200 |
| LastLookAgent | 2.9197 | 29.4575 | 0.0991 | -43.4527 | -61.6934 | 1.1800 |
| TD3Agent | 2.4826 | 30.4361 | 0.0816 | -37.1731 | -73.6635 | 1.1200 |
| NoiseTraderNormal | 1.4215 | 37.1123 | 0.0383 | -65.7972 | -91.8014 | 1.2900 |
| InventorySpreadScalerAgent | 1.0820 | 35.9123 | 0.0301 | -57.0293 | -97.9587 | 1.0800 |
| LSTMSACAgent | 0.9825 | 35.8231 | 0.0274 | -50.1586 | -68.0462 | 1.3400 |
| SACAgent | -0.2717 | 30.2103 | -0.0090 | -68.4734 | -83.4098 | 1.2700 |
| LSTMPPOAgent | -0.7849 | 31.2129 | -0.0251 | -40.2124 | -87.3987 | 1.1800 |
| ASClosedFormAgent | -2.8642 | 30.9023 | -0.0927 | -54.8811 | -71.2641 | 1.4700 |
| PPOAgent | -5.9687 | 39.6030 | -0.1507 | -59.3169 | -118.2767 | 1.2900 |

### GBMRegimeEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| TD3Agent | 9.2689 | 37.7416 | 0.2456 | -35.2118 | -52.2837 | 1.7200 |
| LSTMSACAgent | 8.8750 | 36.5448 | 0.2429 | -37.3646 | -61.4281 | 0.9700 |
| LSTMPPOAgent | 7.6051 | 52.2927 | 0.1454 | -36.8591 | -85.2798 | 1.6400 |
| InventoryShiftAgent | 6.1661 | 38.3061 | 0.1610 | -41.8313 | -63.7419 | 1.0900 |
| SACAgent | 6.0406 | 42.4071 | 0.1424 | -42.9983 | -91.6761 | 1.1900 |
| ZeroIntelligenceAgent | 6.0010 | 39.0047 | 0.1539 | -52.5763 | -90.4834 | 1.1700 |
| PPOAgent | 4.9052 | 36.4974 | 0.1344 | -52.1784 | -63.3313 | 1.2200 |
| ASClosedFormAgent | 4.7075 | 55.4343 | 0.0849 | -63.4827 | -90.1096 | 1.5000 |
| ASSimpleHeuristicAgent | 4.6680 | 22.6321 | 0.2063 | -21.0150 | -47.9458 | 0.8000 |
| MidPriceFollowAgent | 4.1462 | 39.5405 | 0.1049 | -37.1621 | -99.2751 | 1.0900 |
| MarketOrderOnlyAgent | 3.7266 | 24.6889 | 0.1509 | -39.4725 | -54.3119 | 0.8600 |
| InventorySpreadScalerAgent | 2.2959 | 33.9714 | 0.0676 | -52.0676 | -93.0577 | 1.1300 |
| LastLookAgent | 1.9100 | 31.0043 | 0.0616 | -53.5412 | -72.3131 | 1.1100 |
| NoiseTraderUniform | 1.1947 | 33.6460 | 0.0355 | -53.9118 | -84.4283 | 1.0900 |
| FixedSpreadAgent | 0.5426 | 24.7423 | 0.0219 | -44.3849 | -58.9943 | 1.1600 |
| DeepPPOAgent | 0.2184 | 33.4137 | 0.0065 | -64.5554 | -80.8124 | 0.9500 |
| NoiseTraderNormal | -3.1402 | 37.7231 | -0.0832 | -66.9979 | -108.1796 | 1.0600 |

### GBMVanillaEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| InventoryShiftAgent | 6.0395 | 18.7049 | 0.3229 | -17.6654 | -33.5191 | 1.0900 |
| ASSimpleHeuristicAgent | 5.8293 | 17.5556 | 0.3320 | -13.4867 | -27.2492 | 0.9000 |
| MidPriceFollowAgent | 5.1091 | 25.9508 | 0.1969 | -31.3156 | -48.5730 | 1.1900 |
| TD3Agent | 4.8120 | 23.7551 | 0.2026 | -36.2687 | -49.0610 | 1.3000 |
| InventorySpreadScalerAgent | 3.2628 | 16.8966 | 0.1931 | -22.7079 | -32.3258 | 1.0400 |
| ZeroIntelligenceAgent | 2.8495 | 17.4755 | 0.1631 | -24.4541 | -38.0961 | 1.0700 |
| MarketOrderOnlyAgent | 2.4014 | 21.7696 | 0.1103 | -27.5979 | -40.6504 | 1.1300 |
| ASClosedFormAgent | 2.3772 | 28.4385 | 0.0836 | -41.4098 | -62.8452 | 1.5600 |
| FixedSpreadAgent | 2.0400 | 16.7075 | 0.1221 | -20.8830 | -32.4261 | 1.0900 |
| SACAgent | 1.9134 | 21.1694 | 0.0904 | -28.3302 | -49.4841 | 1.0700 |
| NoiseTraderNormal | 1.1711 | 18.7348 | 0.0625 | -22.5520 | -38.9793 | 0.9700 |
| NoiseTraderUniform | 0.8395 | 20.1027 | 0.0418 | -24.9570 | -56.6639 | 1.0900 |
| LastLookAgent | 0.5601 | 23.2506 | 0.0241 | -38.2138 | -58.8418 | 1.1000 |
| PPOAgent | -0.0074 | 18.0070 | -0.0004 | -32.1413 | -44.6132 | 1.0200 |
| LSTMPPOAgent | -0.2047 | 28.0931 | -0.0073 | -26.8484 | -70.3676 | 1.4600 |
| DeepPPOAgent | -0.2712 | 16.2674 | -0.0167 | -26.6237 | -41.7528 | 0.9400 |
| LSTMSACAgent | -0.2955 | 21.0515 | -0.0140 | -32.8843 | -55.9323 | 0.9700 |

### OUJumpEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| NoiseTraderUniform | 4.9233 | 15.1168 | 0.3257 | -15.6542 | -27.8426 | 1.1100 |
| InventoryShiftAgent | 4.4616 | 19.3284 | 0.2308 | -26.3080 | -42.2668 | 1.1700 |
| PPOAgent | 4.2626 | 21.7686 | 0.1958 | -27.2788 | -35.3455 | 1.6800 |
| TD3Agent | 4.0356 | 26.6666 | 0.1513 | -34.9008 | -46.4563 | 1.7000 |
| LastLookAgent | 3.8091 | 19.3689 | 0.1967 | -19.5774 | -33.4784 | 1.2100 |
| FixedSpreadAgent | 3.0988 | 14.1464 | 0.2191 | -19.1185 | -26.7257 | 1.1800 |
| InventorySpreadScalerAgent | 2.8672 | 19.1727 | 0.1495 | -24.6030 | -36.6446 | 1.2000 |
| ZeroIntelligenceAgent | 2.2928 | 16.8505 | 0.1361 | -24.9666 | -40.0452 | 1.2000 |
| MarketOrderOnlyAgent | 1.8248 | 21.0410 | 0.0867 | -24.4631 | -54.5341 | 0.9100 |
| LSTMPPOAgent | 1.7317 | 27.2489 | 0.0636 | -42.2912 | -63.3958 | 2.0300 |
| DeepPPOAgent | 1.5015 | 16.0360 | 0.0936 | -18.1099 | -35.7682 | 1.2600 |
| SACAgent | 1.4181 | 16.1330 | 0.0879 | -25.2084 | -41.2063 | 1.3100 |
| ASSimpleHeuristicAgent | 1.3192 | 11.9146 | 0.1107 | -19.1843 | -30.3763 | 0.8600 |
| NoiseTraderNormal | 0.2007 | 17.1642 | 0.0117 | -23.9227 | -37.4284 | 1.1600 |
| ASClosedFormAgent | 0.1990 | 18.0418 | 0.0110 | -30.6938 | -46.2118 | 1.1500 |
| LSTMSACAgent | 0.1290 | 23.7807 | 0.0054 | -43.8135 | -55.3402 | 1.6100 |
| MidPriceFollowAgent | -0.0828 | 18.0783 | -0.0046 | -27.7926 | -42.8514 | 1.2800 |

### OUJumpRegimeEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| MarketOrderOnlyAgent | 6.5076 | 20.7143 | 0.3142 | -17.6617 | -28.1998 | 1.0400 |
| ASClosedFormAgent | 5.3669 | 16.2876 | 0.3295 | -19.2261 | -28.7865 | 1.2000 |
| LastLookAgent | 4.9509 | 16.3743 | 0.3024 | -22.2241 | -29.1162 | 1.0800 |
| NoiseTraderNormal | 4.1282 | 19.8452 | 0.2080 | -24.5549 | -36.4817 | 0.9500 |
| ASSimpleHeuristicAgent | 3.9933 | 18.7367 | 0.2131 | -23.7883 | -40.3603 | 0.8700 |
| ZeroIntelligenceAgent | 3.1671 | 20.8527 | 0.1519 | -30.2711 | -44.8213 | 1.0400 |
| InventoryShiftAgent | 2.8537 | 18.0097 | 0.1585 | -22.1687 | -36.9832 | 0.9900 |
| FixedSpreadAgent | 2.8357 | 19.3276 | 0.1467 | -22.4012 | -42.4461 | 0.9200 |
| PPOAgent | 2.1603 | 24.9543 | 0.0866 | -33.4895 | -65.6226 | 1.2000 |
| InventorySpreadScalerAgent | 1.6489 | 17.3801 | 0.0949 | -28.4018 | -36.7672 | 1.1300 |
| LSTMSACAgent | 1.3210 | 20.4082 | 0.0647 | -34.2458 | -58.1878 | 1.2600 |
| DeepPPOAgent | 0.7390 | 24.9094 | 0.0297 | -34.0240 | -66.7717 | 1.1400 |
| NoiseTraderUniform | 0.4374 | 16.8699 | 0.0259 | -27.9008 | -36.1111 | 1.0700 |
| LSTMPPOAgent | 0.0736 | 16.0103 | 0.0046 | -23.4639 | -47.9010 | 0.8500 |
| MidPriceFollowAgent | -0.1594 | 18.5557 | -0.0086 | -29.1027 | -42.1908 | 1.0100 |
| TD3Agent | -1.7046 | 24.6935 | -0.0690 | -38.3171 | -68.3281 | 1.6100 |
| SACAgent | -2.3083 | 29.8339 | -0.0774 | -42.0265 | -72.6298 | 1.7200 |

### OURegimeEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| LastLookAgent | 5.1719 | 26.2676 | 0.1969 | -25.1577 | -46.5261 | 1.1500 |
| NoiseTraderUniform | 4.9235 | 23.5839 | 0.2088 | -25.4348 | -42.6519 | 1.2100 |
| SACAgent | 3.5686 | 21.2315 | 0.1681 | -27.8901 | -48.2951 | 1.3700 |
| NoiseTraderNormal | 2.5697 | 22.9638 | 0.1119 | -31.0677 | -54.0791 | 1.2100 |
| FixedSpreadAgent | 2.5412 | 18.0041 | 0.1411 | -20.3190 | -38.1520 | 0.9800 |
| TD3Agent | 1.6197 | 23.8428 | 0.0679 | -32.9938 | -52.1815 | 1.0100 |
| MarketOrderOnlyAgent | 1.4309 | 26.0902 | 0.0548 | -36.4649 | -59.8717 | 1.2900 |
| ASClosedFormAgent | 1.3475 | 19.8651 | 0.0678 | -27.3002 | -47.8858 | 0.9800 |
| PPOAgent | 0.9969 | 23.2973 | 0.0428 | -34.1998 | -56.1503 | 1.5600 |
| LSTMSACAgent | 0.4353 | 21.8163 | 0.0200 | -38.1789 | -56.5866 | 1.3200 |
| ZeroIntelligenceAgent | 0.3943 | 20.9384 | 0.0188 | -27.9159 | -43.9565 | 1.1600 |
| InventorySpreadScalerAgent | -0.2487 | 19.2601 | -0.0129 | -30.5779 | -49.7012 | 1.1100 |
| LSTMPPOAgent | -0.3677 | 32.7814 | -0.0112 | -55.9480 | -75.0435 | 2.6200 |
| ASSimpleHeuristicAgent | -0.6091 | 18.2566 | -0.0334 | -35.7839 | -44.5910 | 0.7300 |
| MidPriceFollowAgent | -1.1404 | 20.4446 | -0.0558 | -37.7355 | -56.2987 | 1.1000 |
| InventoryShiftAgent | -1.3200 | 23.1406 | -0.0570 | -36.9330 | -63.9433 | 0.9500 |
| DeepPPOAgent | -2.3581 | 23.4513 | -0.1006 | -40.1070 | -70.9946 | 1.4500 |

### OUVanillaEnv

| Agent | Mean | Std | Sharpe | VaR (95%) | ES (95%) | Avg Inventory |
|-------|------|-----|--------|-----------|----------|---------------|
| SACAgent | 4.7365 | 22.8957 | 0.2069 | -21.1790 | -37.2103 | 1.2200 |
| ASClosedFormAgent | 3.9560 | 19.4509 | 0.2034 | -30.4728 | -41.5856 | 1.1900 |
| LSTMSACAgent | 3.6185 | 22.4972 | 0.1608 | -25.7127 | -57.3637 | 1.6300 |
| NoiseTraderNormal | 3.5206 | 19.7483 | 0.1783 | -26.3764 | -42.6682 | 1.4200 |
| ASSimpleHeuristicAgent | 2.6853 | 14.5644 | 0.1844 | -16.1225 | -25.6496 | 0.8800 |
| FixedSpreadAgent | 2.4388 | 15.5138 | 0.1572 | -22.6825 | -30.2284 | 1.0500 |
| MidPriceFollowAgent | 2.3505 | 15.6762 | 0.1499 | -20.6888 | -39.4764 | 1.0300 |
| InventoryShiftAgent | 2.3391 | 14.9586 | 0.1564 | -20.0860 | -26.2819 | 1.0600 |
| TD3Agent | 2.1703 | 14.5537 | 0.1491 | -18.7583 | -29.8735 | 1.1700 |
| ZeroIntelligenceAgent | 2.0824 | 18.6577 | 0.1116 | -25.9484 | -42.2122 | 1.3100 |
| NoiseTraderUniform | 1.9930 | 17.2384 | 0.1156 | -24.6405 | -30.1460 | 1.2100 |
| DeepPPOAgent | 1.7562 | 19.4756 | 0.0902 | -18.0730 | -49.1430 | 1.0100 |
| MarketOrderOnlyAgent | 1.6316 | 19.0634 | 0.0856 | -27.4668 | -39.1614 | 0.9900 |
| LastLookAgent | 1.4471 | 16.4055 | 0.0882 | -20.4373 | -35.7193 | 1.0200 |
| PPOAgent | 1.3792 | 15.0832 | 0.0914 | -19.0296 | -29.4213 | 1.2600 |
| LSTMPPOAgent | 1.2380 | 22.2071 | 0.0557 | -31.6211 | -51.2400 | 1.6200 |
| InventorySpreadScalerAgent | 0.7338 | 17.4474 | 0.0421 | -23.0248 | -36.0536 | 1.1600 |

## Key Insights and Observations

1. **Best Overall Performance:** FixedSpreadAgent achieves the highest mean PnL (10.0936) on ABMJumpRegimeEnv.

2. **Most Consistent Agent:** ASSimpleHeuristicAgent has the lowest average standard deviation (18.0747), indicating more stable performance.

3. **Best Risk-Adjusted Return:** DeepPPOAgent achieves the highest Sharpe ratio (0.3479) on ABMRegimeEnv.

4. **Agent Category Comparison:**
   - RL Agents average mean PnL: 2.0827
   - Analytic Agents average mean PnL: 2.6700
   - Heuristic Agents average mean PnL: 2.4562

5. **Environment Difficulty:**
   - Most challenging environment: OURegimeEnv (avg mean PnL: 1.1150)
   - Easiest environment: GBMRegimeEnv (avg mean PnL: 4.0666)

---

*Report generated from existing comparison results.*
