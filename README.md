# Final Code - 66.5% Leaderboard Submission

This folder contains all the code and files needed to:
1. retrain/tune the model pipeline.
2. regenerate the final submitted CSV.

Files in this folder:
- `train_tune.py` (tuning/training pipeline)
- `make_submission.py` (submission script using a JSON parameter file)
- `make_submission_shift_tta.py` (final shift-TTA submission generator)
- `params_final.json` (the tuned parameters used for the final submission)

## Important
Run commands from the **repository root**, the folder containing `Xtr.csv`, `Ytr.csv`, `Xte.csv`.

## A) Regenerate the exact final submitted model CSV:
This reproduces the final approach used for the best leaderboard result:

```bash
python final_code/make_submission_shift_tta.py --mode submit --json final_code/params_final.json --shift-weight 1.0 --output final_code/submission_final_66p5_style.csv
```

This uses:
- tuned JSON parameters  (`params_final.json`) 
- fixed shift test-time augmentation averaging (`shift_weight=1.0`)

## B) Retune/train from scratch, then submit
If needed, retune parameters:

```bash
python final_code/train_tune.py --mode validate --params final_code/params_regenerated.json
```

Then generate submission from the regenerated JSON:

```bash
python final_code/make_submission_shift_tta.py --mode submit --json final_code/params_regenerated.json --shift-weight 1.0 --output final_code/submission_from_regenerated_params.csv
```

# Notes on reproducibility

- The command in Section A should reproduce the final submission exactly, since it uses the provided tuned JSON file and fixed test-time augmentation settings.

- If you retune from scratch as in Section B, you should get very similar results, but they may not be perfectly identical on every machine. This is because small numerical differences can appear across environments and linear algebra backends.

