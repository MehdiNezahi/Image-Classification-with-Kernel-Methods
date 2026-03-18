# EXPLAIN_CODE.md

This file explains the **code logic** of the final folder: what each script does, how functions connect, and the execution flow from training to submission.

Folder:
- `train_tune.py`
- `make_submission.py`
- `make_submission_shift_tta.py`
- `params_final.json`

---

## 1. Overall architecture

There are two layers:

1. **Core model layer** (`train_tune.py` + shared functions used by `make_submission.py`)
- load data
- extract features
- build grouped kernels
- train OVR kernel ridge
- get class scores

2. **Submission layer**
- `make_submission.py`: submit with tuned JSON parameters
- `make_submission_shift_tta.py`: same tuned model, plus shift-TTA ensemble before final prediction

---

## 2. `train_tune.py` logic

This is the main training/tuning script.

## 2.1 Top-level constants

The script starts with config constants:
- split settings (`VAL_RATIO`, `SPLIT_SEED`)
- search grids (`WEIGHT_GRID`, `GAMMA_GRID`, `REG_GRID`)
- search budget (`N_TRIALS_*`, `COORD_PASSES`)
- caching settings (`CACHE_DISTANCES`, `MAX_CACHE_GB`)
- baseline params (`BASE_WEIGHTS`, `BASE_GAMMA_MULT`, `BASE_REG`)
- fixed classwise flip blend (`CLASS_ALPHA`)

These constants control reproducibility and search cost.

## 2.2 Data and transform helpers

Functions:
- `load_X`, `load_y`
- `stratified_train_val_split`
- `transform_vectors` (orig / hflip)

Role:
- load data in memory-efficient chunks,
- produce deterministic stratified split,
- provide hflip view for TTA-style score fusion.

## 2.3 Feature extractor functions

Functions include:
- block reducers (`block_reduce_from_images`, `block_reduce_single_channel`)
- gradient operators (`central_diff_x`, `central_diff_y`)
- descriptors (`pooled_hog`, `census_texture`, `laplace_energy`, `opp_grad_features`, `orient_energy_features`)
- group assembly (`build_groups_69`)

Logic:
- all handcrafted features are computed per image,
- returned as a dict of named groups, each group = 2D feature matrix `(n_samples, n_features_g)`.

## 2.4 Standardization and kernel utilities

Functions/classes:
- `Standardizer`
- `standardize_groups`, `apply_standardizers`
- `squared_distances`, `rbf_kernel_from_distances`, `heuristic_gamma`
- `center_kernel_train`, `center_kernel_test`, `normalize_kernel_pair`
- `build_kernel_pair`

Logic:
- standardize each feature group independently,
- compute per-group RBF kernels with tuned gamma multipliers,
- center/normalize each kernel,
- sum weighted kernels into one final train/eval kernel pair.

## 2.5 Classifier

Class:
- `KernelRidgeOVR`

Methods:
- `fit_from_precomputed_kernel`
- `score`

Logic:
- solve one linear system for all classes at once,
- return class score matrix for eval samples.

## 2.6 Caching for speed

Functions:
- `estimate_cache_bytes`
- `maybe_precompute_distances`

Logic:
- if estimated memory is small enough, precompute distance matrices once and reuse during hyperparameter trials.
- reduces repeated distance computation cost.

## 2.7 Validation objective wrapper

Function:
- `evaluate_config`

What it does:
1. build kernels for current `(weights, gamma_mult, reg)`,
2. train OVR kernel ridge,
3. score orig and hflip,
4. combine scores with fixed `CLASS_ALPHA`,
5. compute validation accuracy.

This is the objective used by the tuner.

## 2.8 Hyperparameter tuner

Function:
- `tune_hyperparameters`

Search flow:
1. evaluate baseline config
2. tune `reg` on grid
3. random search over weights
4. random search over gamma multipliers
5. coordinate ascent passes:
   - reg refinement
   - one-group weight refinement
   - one-group gamma refinement

Returns:
- best `weights`, `gamma_mult`, `reg`, `acc`.

## 2.9 Parameter persistence

Functions:
- `save_params`
- `load_params`

Logic:
- JSON output allows reusing tuned parameters in submission scripts.

## 2.10 Scoring with fixed params

Function:
- `compute_scores_with_params`

Logic:
- rebuild full pipeline once with chosen params,
- return fused class scores for any eval set (val/test).

## 2.11 Entry points

Functions:
- `run_validate`
- `run_submit`
- `main` with modes `validate`, `submit`, `both`.

Usage:
- `validate`: tune and save JSON
- `submit`: load JSON (or tune if missing), train full model, write CSV

---

## 3. `make_submission.py` logic

This script is a cleaner deployment path for JSON-driven submission.

## 3.1 Core behavior

Flow in `run_submit_tuned_from_json`:
1. load tuned JSON (`load_tuned_params`)
2. convert params into model cfg (`make_cfg_from_tuned_params`)
3. load full train and test
4. compute orig/hflip scores (`fit_model_scores_decomposed`)
5. fuse:
   - classwise alpha from JSON if available
   - else 0.5/0.5
6. argmax and write submission CSV

## 3.2 Important implementation note

`class_model` is parsed from JSON if present, but current logic does not use it for prediction routing.  
Prediction fusion uses `class_alpha` (or 0.5/0.5 fallback).

---

## 4. `make_submission_shift_tta.py` logic

This script wraps the tuned model with shift-based TTA.

## 4.1 Additional functions

- `shift_vectors`: applies small cyclic translation via `np.roll`.
- `score_view`: gets tuned-model class scores for one input view.

## 4.2 Validation mode

`run_validate`:
1. compute base scores (original pipeline)
2. compute scores on shifted views
3. average shifted-view scores
4. test blend `S = (1-w)*S_base + w*S_shift_avg` over several `w`
5. print best `w` on split-42 validation

## 4.3 Submit mode

`run_submit`:
1. train tuned model on full train
2. score test on base + shifts
3. build final score with chosen `shift_weight`
4. write submission CSV

In your final usage, `shift_weight=1.0` was used (pure shift-average).

---

## 5. `params_final.json` logic role

This file is the frozen parameter state used for deterministic regeneration:
- group kernel weights
- gamma multipliers
- regularization
- optional score-fusion metadata

It separates optimization from deployment:
- no retuning needed to regenerate final submission.

---

## 6. End-to-end execution paths

## 6.1 Fast regeneration of final submission style

Use precomputed JSON:
```bash
python final_code/make_submission_shift_tta.py --mode submit --json final_code/params_final.json --shift-weight 1.0 --output final_code/submission_final_66p5_style.csv
```

## 6.2 Full retune then submission

```bash
python final_code/train_tune.py --mode validate --params final_code/params_regenerated.json
python final_code/make_submission_shift_tta.py --mode submit --json final_code/params_regenerated.json --shift-weight 1.0 --output final_code/submission_from_regenerated_params.csv
```

---

## 7. Practical code-design rationale

Why this code structure is effective:
- clear separation between tuning and deployment
- deterministic split and search settings
- reusable feature/kernel primitives
- JSON-based parameter checkpointing
- optional augmentation wrapper script that does not alter base training code

This design makes verification by your professor straightforward:
- inspect code,
- run command,
- regenerate submission from provided parameter JSON.

