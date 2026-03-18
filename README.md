# Final Code (66.5% LB Submission)

This folder contains the exact code/assets needed to:
1. retrain/tune the model pipeline, and
2. regenerate the final submitted CSV.

Files in this folder:
- `train_tune.py` (tuning/training pipeline)
- `make_submission.py` (JSON-driven submission pipeline)
- `make_submission_shift_tta.py` (final shift-TTA submission generator)
- `params_final.json` (the tuned parameters used for the final submission)

## Important
Run commands from the **repository root** (the folder containing `Xtr.csv`, `Ytr.csv`, `Xte.csv`).

## A) Regenerate the exact final submitted model CSV (recommended)
This reproduces the final approach used for the best leaderboard result:

```bash
python final_code/make_submission_shift_tta.py --mode submit --json final_code/params_final.json --shift-weight 1.0 --output final_code/submission_final_66p5_style.csv
```

This uses:
- tuned JSON parameters (`params_final.json`)
- fixed shift test-time augmentation averaging (`shift_weight=1.0`)

## B) Retune/train from scratch, then submit
If needed, retune parameters on the fixed local split protocol (80/20 stratified, seed 42):

```bash
python final_code/train_tune.py --mode validate --params final_code/params_regenerated.json
```

Then generate submission from the regenerated JSON:

```bash
python final_code/make_submission_shift_tta.py --mode submit --json final_code/params_regenerated.json --shift-weight 1.0 --output final_code/submission_from_regenerated_params.csv
```

## Notes on reproducibility
- The exact final submission regeneration (Section A) is deterministic given the provided JSON and fixed augmentation settings.
- The scratch retuning path (Section B) uses fixed seeds in code but may still vary slightly across environments/BLAS backends.

