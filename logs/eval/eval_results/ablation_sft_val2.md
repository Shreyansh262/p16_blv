# BLV SFT Ablation Study — sft_val2 vs Base Model
Generated: 2026-03-24

## 1. Teacher Data Audit
| Metric | Score |
|--------|-------|
| Total samples | 9,646 |
| After quality filter (>30 chars) | 9,646 (100%) |
| Spatial terms (left/right/ahead…) | 87/100 |
| Room identification | 66/100 |
| Hazard awareness | 78/100 |
| Height references | 80/100 |
| Distance measurements | 13/100 |
| Samples with 3+ of 4 BLV props | 73/100 |
| Caption length (median words) | 53 |

**Verdict:** Teacher data quality is acceptable. Not the bottleneck.
Spatial, hazard, room coverage all pass thresholds. Distance measurements (13%) are weak but secondary.

## 2. Dataset BLV Score Distribution (all 9,646 samples)
| Bucket | Count | % |
|--------|-------|---|
| poor(0-24) | 7 | 0.1% |
| fair(25-49) | 1,090 | 11.3% |
| good(50-74) | 4,295 | 44.5% |
| excellent(75+) | 4,254 | 44.1% |
| **Average** | **69.5/100** | |

**Implication:** 44% of data is excellent (≥75). Using top-scoring 3,000 samples for sft_val3 is safe.

## 3. sft_val2 Training Summary
| Parameter | Value |
|-----------|-------|
| Training samples | ~1,445 (15% of 9,646) |
| Epochs | 3 |
| Learning rate | 1e-5 |
| LoRA rank | 64 |
| Epoch 1 loss | 13.2729 |
| Epoch 2 loss | 0.2540 |
| Epoch 3 loss | 0.0813 |
| Verdict | 🔴 OVERFIT — epoch3 loss < 0.5 |

## 4. Ablation Results — BLV Score on 10 Golden Samples
| video_id | dataset | Reference | Base | sft_val2 | Delta (base→ft) |
|----------|---------|-----------|------|----------|-----------------|
| YT43U | charades | 86 | 34 | 38 | +4 |
| QXCUP | charades | 86 | 30 | 32 | +2 |
| FS3SY | charades | 92 | 58 | 10 | -48 |
| JJGEU | charades | 92 | 70 | 40 | -30 |
| X1RBM | charades | 90 | 34 | 42 | +8 |
| WL1DJ | charades | 90 | 48 | 38 | -10 |
| LU82W | charades | 92 | 50 | 32 | -18 |
| 2525468588 | avcaps | 82 | 50 | 40 | -10 |
| 13090056545 | avcaps | 96 | 42 | 36 | -6 |
| 7040233679 | avcaps | 92 | 58 | 32 | -26 |
| **AVERAGE** | | **89.8** | **47.4** | **34.0** | **-13.4** |

### Key Findings
- **sft_val2 is WORSE than base model** by 13.4 BLV points (avg)
- 7/10 samples regressed after fine-tuning
- Only 3/10 improved: YT43U (+4), QXCUP (+2), X1RBM (+8)
- Worst regression: FS3SY (base=58 → ft=10, Δ=-48) — model output was 'The text does not provide enough information'
- Both models are far from reference (gap to ref: base=+42.4, ft=+55.8)

## 5. Root Cause Analysis
| Root Cause | Evidence | Fix in sft_val3 |
|------------|----------|-----------------|
| Overfitting (too few samples) | epoch3_loss=0.08, 7/10 regressions | Increase to 3,000 top-quality samples |
| Too many epochs | Loss 13→0.08 in 3 epochs = too fast | Cap at 2 epochs |
| No quality gate on training data | Random 15% sample (not BLV-scored) | Select top-3000 by BLV score |
| No BLV-aware stopping criterion | Only eval_loss used | Add BLV callback every 100 steps |
| Mode collapse on some samples | FS3SY: 'text does not provide enough info' | Add no_repeat_ngram_size=3 |

## 6. sft_val3 Configuration (Planned)
```python
max_samples          = 3000   # top-scoring by BLV score (vs random 15%)
num_train_epochs     = 2      # was 3 — stop before epoch3 collapse
learning_rate        = 1e-5   # unchanged — was already correct
warmup_steps         = 50     # unchanged
early_stopping_patience = 3  # NEW: stop if eval_loss doesn't improve
no_repeat_ngram_size = 3      # NEW: prevent mode collapse
blv_score_eval_steps = 100   # NEW: log BLV score every 100 steps
stop_if_blv_plateau  = True  # NEW: stop if BLV score flat for 300 steps
data_selection       = 'top_blv_score'  # NEW: was random
```

## 7. Expected Trajectory for sft_val3
| Checkpoint | Est. Loss | Target BLV Score | Interpretation |
|------------|-----------|------------------|----------------|
| Base model | — | ~47.4 | Measured baseline |
| Step 100 | ~6-8 | ~50-55 | Starting to learn BLV structure |
| Step 300 | ~3-5 | ~55-65 | Spatial terms emerging |
| Step 500 | ~1.5-3 | ~60-70 | Consistent room ID + hazard |
| Step 800+ | <1.5 | <60 plateau | Stop here if BLV score plateaus |
| Target (2 epochs) | ~1.5-2.5 | ≥65 | Minimum bar for proceed to full run |

## 8. Stop/Go Criteria for sft_val3
- **GO**: avg BLV score on golden set ≥ 65/100 at any checkpoint
- **STOP/REPLAN**: BLV score never exceeds 55 → data selection strategy is wrong
- **STOP/OVERFIT**: BLV score peaks then drops while loss keeps falling
- **ABORT**: Any sample outputs 'text does not provide' / repetition loops
