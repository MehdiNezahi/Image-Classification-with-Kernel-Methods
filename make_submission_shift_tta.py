import argparse
import numpy as np
import pandas as pd

from make_submission import (
    load_X,
    load_y,
    transform_vectors,
    load_tuned_params,
    make_cfg_from_tuned_params,
    fit_model_scores_decomposed,
)
from train_tune import stratified_train_val_split


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def shift_vectors(X, dx=0, dy=0):
    n = X.shape[0]
    imgs = X.reshape(n, 3, 32, 32).astype(np.float32)
    out = np.empty_like(imgs)
    for i in range(n):
        for c in range(3):
            out[i, c] = np.roll(imgs[i, c], shift=(dy, dx), axis=(0, 1))
    return out.reshape(n, -1).astype(np.float32)


def compose_scores(So, Sh, class_alpha):
    if class_alpha is None:
        return 0.5 * So + 0.5 * Sh
    S = np.empty_like(So, dtype=np.float64)
    for c in range(So.shape[1]):
        a = float(class_alpha[c])
        S[:, c] = a * So[:, c] + (1.0 - a) * Sh[:, c]
    return S


def score_view(Xtr, ytr, Xe, cfg, class_alpha):
    Xeh = transform_vectors(Xe, "hflip")
    So, Sh = fit_model_scores_decomposed(Xtr, ytr, Xe, Xeh, cfg)
    return compose_scores(So, Sh, class_alpha)


def run_validate(json_path):
    weights, gamma_mult, reg, class_alpha, _ = load_tuned_params(json_path)
    cfg = make_cfg_from_tuned_params(weights, gamma_mult, reg)
    X = load_X("Xtr.csv")
    y = load_y("Ytr.csv")
    Xtr, ytr, Xv, yv = stratified_train_val_split(X, y, val_ratio=0.2, seed=42)

    S_base = score_view(Xtr, ytr, Xv, cfg, class_alpha)
    base_acc = accuracy(yv, np.argmax(S_base, axis=1))
    print(f"Base acc: {100*base_acc:.2f}%")

    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    scores = []
    for dx, dy in shifts:
        Xs = Xv if (dx == 0 and dy == 0) else shift_vectors(Xv, dx=dx, dy=dy)
        S = score_view(Xtr, ytr, Xs, cfg, class_alpha)
        scores.append(S)
    S_avg = np.mean(np.stack(scores, axis=0), axis=0)
    acc_avg = accuracy(yv, np.argmax(S_avg, axis=1))
    print(f"Shift-avg (base+4 shifts) acc: {100*acc_avg:.2f}%")

    best = (base_acc, 0.0)
    for w in np.linspace(0.0, 1.0, 11):
        S = (1.0 - w) * S_base + w * S_avg
        acc = accuracy(yv, np.argmax(S, axis=1))
        print(f"  shift_weight={w:.2f} -> {100*acc:.2f}%")
        if acc > best[0] + 1e-12:
            best = (acc, float(w))
    print(f"Best mixed acc={100*best[0]:.2f}% at shift_weight={best[1]:.2f}")


def run_submit(json_path, output_csv, shift_weight=0.5):
    weights, gamma_mult, reg, class_alpha, _ = load_tuned_params(json_path)
    cfg = make_cfg_from_tuned_params(weights, gamma_mult, reg)
    Xtr = load_X("Xtr.csv")
    ytr = load_y("Ytr.csv")
    Xte = load_X("Xte.csv")

    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    scores = []
    for dx, dy in shifts:
        Xs = Xte if (dx == 0 and dy == 0) else shift_vectors(Xte, dx=dx, dy=dy)
        S = score_view(Xtr, ytr, Xs, cfg, class_alpha)
        scores.append(S)
    S_avg = np.mean(np.stack(scores, axis=0), axis=0)
    S_base = scores[0]
    S = (1.0 - shift_weight) * S_base + shift_weight * S_avg
    pred = np.argmax(S, axis=1).astype(np.int64)
    sub = pd.DataFrame({"Id": np.arange(1, Xte.shape[0] + 1), "Prediction": pred})
    sub.to_csv(output_csv, index=False)
    print(f"Saved submission: {output_csv}")
    print(f"Used shift_weight={shift_weight}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "submit"], default="validate")
    parser.add_argument("--json", type=str, default="tuned_params_mixed_kernel.json")
    parser.add_argument("--shift-weight", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="submission_69p2_tuned_shift_quick.csv")
    args = parser.parse_args()

    if args.mode == "validate":
        run_validate(args.json)
    else:
        run_submit(args.json, args.output, shift_weight=args.shift_weight)


if __name__ == "__main__":
    main()
