# LuckyCharm

Behavioral analysis pipeline for mouse decision-making experiments. Processes raw event logs through quality control, trial extraction, lick detection, and statistical modeling to study waiting behavior and reward learning.

## Setup

```bash
conda env create -f environment.yml
conda activate LUCKYCHARM
```

## Pipeline

Run scripts in order:

| Step | Script | Description |
|------|--------|-------------|
| 0 | `0_session_quality_control.py` | Validate and back up raw sessions |
| 1 | `1_data_set_curation.ipynb` | Identify missing sessions |
| 2 | `2_events_processing.py` | Parse raw behavioral events |
| 3 | `3_events_stitching.py` | Combine multi-session recordings |
| 4 | `4_lick_detection.py` | Detect lick bouts and artifacts |
| 5 | `5_trials_analysis.py` | Extract trial-level features |
| 6 | `6_data_cleaning.ipynb` | Filter and clean trial data |
| 7 | `7_LMEM.ipynb` | Linear mixed-effects modeling |
| 8 | `8_HMM.py` | Hidden Markov Model state classification |

## Analysis

- `delta_t_analysis.py` — Trial-to-trial ΔT regression analysis
- `train_models_by_group.py` — Random Forest models per cohort group
- `visualize_rf_results.py` — Feature importance and model performance plots
- `timing_anchor_cv.ipynb` — Cross-validation of timing reference points

## Config

`exp_cohort_info.json` maps mouse IDs to experiments and cohorts. Update this when adding new animals.
