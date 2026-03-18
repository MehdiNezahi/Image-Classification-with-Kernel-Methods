# EXPLAIN_METHOD.md

This document explains, in full technical detail, the method in `final_code/` used to produce the final submission pipeline.

Relevant files:
- `train_tune.py`
- `make_submission.py`
- `make_submission_shift_tta.py`
- `params_final.json`

---

## 1. Goal and constraints

Task:
- 10-class image classification with train/test CSV files.

Challenge constraints (satisfied):
- final classifier must be kernel-based.
- no forbidden external ML libraries.

Model family used:
- One-vs-Rest Kernel Ridge Classification with precomputed grouped RBF kernels.

---

## 2. Data format and split protocol

Input format:
- each sample has 3072 values = flattened `32x32x3`.
- channel layout:
  - `0:1024` -> red-like channel
  - `1024:2048` -> green-like channel
  - `2048:3072` -> blue-like channel

Validation protocol in tuning:
- stratified 80/20 split
- seed = 42

Train/val sizes in tuning:
- train subset: 4000
- validation subset: 1000

This split is used by `train_tune.py` when tuning parameters.

---

## 3. Geometric transforms and channels

## 3.1 Horizontal flip

Function `transform_vectors(X, "hflip")`:
- reshape to `(n,3,32,32)`,
- reverse width axis,
- flatten back.

This is used as TTA view in all final pipelines.

## 3.2 Opponent channel transform

From image channels \(R,G,B\), define:
\[
\text{gray} = \frac{R+G+B}{3},\quad
rg = R-G,\quad
yb = \frac{R+G}{2}-B
\]

This produces intensity and chromatic-opponent representations.

---

## 4. Handcrafted feature extractor (`build_groups_69`)

The final model uses grouped handcrafted descriptors.

## 4.1 Block pooled intensity/opponent statistics

At block sizes \(b\in\{2,4\}\):
- `raw_means`, `raw_stds` from RGB channels
- `opp_means`, `opp_stds` from opponent channels

Pooling operators:
- mean / std (and max in some groups).

## 4.2 Gradient and HOG-like groups

Central differences:
\[
g_x(i,j)=\frac{x(i,j+1)-x(i,j-1)}{2},\quad
g_y(i,j)=\frac{x(i+1,j)-x(i-1,j)}{2}
\]
(with edge handling on borders).

Then:
- magnitude \(m=\sqrt{g_x^2+g_y^2}\)
- orientation \(\theta=\operatorname{mod}(\operatorname{atan2}(g_y,g_x),\pi)\)
- orientation-bin pooled maps over block sizes `(4,8)`.

Produced groups:
- `hog_mean`
- `hog_max`
- `grad_max`
- `local_energy`

## 4.3 Texture and second-order structure groups

- `census`: local binary pattern-like code from 8 neighbors; pooled mean/std/max.
- `laplace`: Laplacian-based absolute and squared energy pooled stats.
- `opp_grad`: HOG-like gradients on `rg` and `yb`.

## 4.4 Directional edge energy group

`orient_energy`:
- for maps in `{gray, rg, yb}`,
- project gradients onto directions \(0,\pi/4,\pi/2,3\pi/4\):
\[
r_\theta = |\cos\theta\cdot g_x + \sin\theta\cdot g_y|
\]
- pool `(mean,max,std)` at `(4,8)`.

This provides explicit directional edge-energy statistics.

---

## 5. Standardization

Each group is standardized independently:
\[
\tilde x = \frac{x-\mu}{\sigma}
\]
where \(\mu,\sigma\) are computed on training subset only.

The same group scalers are applied to validation/test features.

---

## 6. Grouped kernel construction

For group \(g\), squared distance matrix:
\[
D_g(i,j)=\|x_i^{(g)}-x_j^{(g)}\|^2
\]

RBF kernel:
\[
K_g(i,j)=\exp(-\gamma_g D_g(i,j))
\]

Gamma definition:
\[
\gamma_g = \gamma_g^{\text{base}}\cdot m_g
\]
where:
- \(\gamma_g^{\text{base}}\) from median heuristic,
- \(m_g\) is tuned group-specific multiplier.

For groups `hog_mean`, `hog_max`, `opp_grad`, fixed two-bandwidth mix:
\[
K_g^{mix}=0.7K_g(\gamma_g)+0.3K_g(2\gamma_g)
\]

Each group kernel is:
1. centered,
2. normalized by mean diagonal scale,
3. weighted by \(w_g\),
4. summed:
\[
K = \sum_g w_g\,\widehat K_g
\]

This gives one final train kernel and one eval/test kernel.

---

## 7. Classifier: OVR Kernel Ridge

For \(n\) train samples and \(C=10\) classes:
- target matrix \(Y\in\{-1,+1\}^{n\times C}\)
- solve:
\[
(K+\lambda I)\alpha = Y
\]

Scores for eval/test:
\[
S = K_{\text{eval,train}}\alpha
\]

Prediction:
\[
\hat y = \arg\max_c S_c
\]

This is implemented in `KernelRidgeOVR`.

---

## 8. Tuning procedure (`train_tune.py`)

The script tunes three parameter families:
- group weights \(w_g\)
- gamma multipliers \(m_g\)
- regularization \(\lambda\)

Search steps:
1. Start from baseline params (`BASE_WEIGHTS`, `BASE_GAMMA_MULT`, `BASE_REG`).
2. Scan regularization over `REG_GRID`.
3. Random search over weight vectors (`N_TRIALS_WEIGHTS`).
4. Random search over gamma vectors (`N_TRIALS_GAMMA`).
5. Coordinate ascent passes (`COORD_PASSES`):
   - reg refinement
   - one-group-at-a-time weight refinement
   - one-group-at-a-time gamma refinement

Scoring objective during tuning:
- validation accuracy on split seed 42.

Saved output JSON:
- `weights`
- `gamma_mult`
- `reg`
- validation score metadata

---

## 9. Distance caching optimization in tuning

To accelerate repeated evaluations:
- script can precompute per-group distance matrices for:
  - train/train
  - val/train
  - val_hflip/train
- only if estimated memory is below threshold (`MAX_CACHE_GB`).

This keeps math identical while reducing recomputation time.

---

## 10. Submission pipeline (`make_submission.py`)

Given tuned JSON params:
1. load `weights`, `gamma_mult`, `reg` (and optionally `class_alpha`).
2. train on full train set (`Xtr`,`Ytr`).
3. score test on original and hflip views.
4. fuse scores:
   - if `class_alpha` exists:
\[
S_c = a_c S^{orig}_c + (1-a_c)S^{flip}_c
\]
   - else:
\[
S = 0.5S^{orig} + 0.5S^{flip}
\]
5. output CSV `Id,Prediction`.

---

## 11. Final augmentation wrapper used for best leaderboard submission (`make_submission_shift_tta.py`)

On top of tuned JSON model:
1. evaluate test-time scores for multiple spatially shifted views:
   - base `(0,0)`,
   - shifts `(1,0),(-1,0),(0,1),(0,-1)` via `np.roll`.
2. each view internally still uses original+hflip fusion.
3. average shifted-view scores (best run used `shift_weight=1.0`, i.e. full shift-average ensemble).
4. argmax over classes and write CSV.

This is a test-time augmentation ensemble that improved robustness and leaderboard score.

---

## 12. Exact reproducibility paths

## 12.1 Reproduce final submitted style (recommended)

Using provided tuned JSON:
```bash
python final_code/make_submission_shift_tta.py --mode submit --json final_code/params_final.json --shift-weight 1.0 --output final_code/submission_final_66p5_style.csv
```

## 12.2 Retune from scratch then submit

```bash
python final_code/train_tune.py --mode validate --params final_code/params_regenerated.json
python final_code/make_submission_shift_tta.py --mode submit --json final_code/params_regenerated.json --shift-weight 1.0 --output final_code/submission_from_regenerated_params.csv
```

---

## 13. Why this method generalized better than many higher local-score variants

Empirically, this pipeline is:
- strong feature-wise (rich handcrafted groups),
- kernel-faithful,
- relatively simple in parameterization compared to heavy classwise multi-model fusion.

That lower complexity can reduce overfitting to one validation split, improving leaderboard transfer.

---

## 14. Compliance checklist

- Kernel method: yes (grouped RBF + OVR kernel ridge).
- Forbidden ML libraries: no.
- Submission format compliant: yes (`Id,Prediction`).
- Deterministic reproduction from provided JSON and fixed augmentation settings: yes.

