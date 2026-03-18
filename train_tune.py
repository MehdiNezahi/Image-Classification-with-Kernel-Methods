import argparse
import json
import os
import numpy as np
import pandas as pd

VAL_RATIO = 0.2
SPLIT_SEED = 42
GAMMA_SUBSET = 700
NUM_CLASSES = 10

WEIGHT_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
GAMMA_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
REG_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
N_TRIALS_WEIGHTS = 80
N_TRIALS_GAMMA = 80
COORD_PASSES = 2
RANDOM_SEED = 123

CACHE_DISTANCES = True
MAX_CACHE_GB = 1.5

FIXED_MIX_GROUPS = {"hog_mean", "hog_max", "opp_grad"}

BASE_WEIGHTS = {
    "raw_means": 0.5,
    "raw_stds": 0.5,
    "opp_means": 1.0,
    "opp_stds": 0.75,
    "hog_mean": 2.5,
    "hog_max": 2.0,
    "grad_max": 1.5,
    "local_energy": 1.0,
    "census": 1.2,
    "laplace": 1.2,
    "opp_grad": 1.2,
    "orient_energy": 0.8,
}

BASE_GAMMA_MULT = {
    "raw_means": 1.0,
    "raw_stds": 0.5,
    "opp_means": 1.0,
    "opp_stds": 0.5,
    "hog_mean": 1.5,
    "hog_max": 2.0,
    "grad_max": 1.5,
    "local_energy": 1.0,
    "census": 1.5,
    "laplace": 1.2,
    "opp_grad": 1.5,
    "orient_energy": 1.2,
}

BASE_REG = 3.0

CLASS_ALPHA = np.array([0.70, 0.35, 0.10, 0.45, 0.75, 0.15, 0.30, 0.50, 0.55, 0.70], dtype=np.float64)

PARAMS_PATH = "tuned_params_69p2.json"


def load_X(path):
    chunks = []
    for ch in pd.read_csv(path, header=None, usecols=range(3072), chunksize=500):
        chunks.append(ch.values.astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def load_y(path):
    return np.array(pd.read_csv(path, usecols=[1])).squeeze().astype(np.int64)


def stratified_train_val_split(X, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_val_c = int(len(idx_c) * val_ratio)
        val_indices.extend(idx_c[:n_val_c])
        train_indices.extend(idx_c[n_val_c:])
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]



def transform_vectors(X, mode):
    n = X.shape[0]
    imgs = X.reshape(n, 3, 32, 32)
    if mode == "orig":
        out = imgs
    elif mode == "hflip":
        out = imgs[:, :, :, ::-1]
    else:
        raise ValueError(mode)
    return out.reshape(n, -1).astype(np.float32)


def vector_to_image_batch(X):
    n = X.shape[0]
    r = X[:, :1024].reshape(n, 32, 32)
    g = X[:, 1024:2048].reshape(n, 32, 32)
    b = X[:, 2048:3072].reshape(n, 32, 32)
    return np.stack([r, g, b], axis=-1)


def opponent_channels(imgs):
    r = imgs[..., 0]
    g = imgs[..., 1]
    b = imgs[..., 2]
    gray = (r + g + b) / 3.0
    rg = r - g
    yb = 0.5 * (r + g) - b
    return np.stack([gray, rg, yb], axis=-1)



def block_reduce_from_images(imgs, block_size, mode="mean"):
    n, h, w, c = imgs.shape
    hb = h // block_size
    wb = w // block_size
    reshaped = imgs.reshape(n, hb, block_size, wb, block_size, c)
    if mode == "mean":
        pooled = reshaped.mean(axis=(2, 4))
    elif mode == "std":
        pooled = reshaped.std(axis=(2, 4))
    elif mode == "max":
        pooled = reshaped.max(axis=(2, 4))
    else:
        raise ValueError(mode)
    return pooled.reshape(n, -1).astype(np.float32)


def block_reduce_single_channel(x, block_size, mode="mean"):
    n, h, w = x.shape
    hb = h // block_size
    wb = w // block_size
    reshaped = x.reshape(n, hb, block_size, wb, block_size)
    if mode == "mean":
        pooled = reshaped.mean(axis=(2, 4))
    elif mode == "std":
        pooled = reshaped.std(axis=(2, 4))
    elif mode == "max":
        pooled = reshaped.max(axis=(2, 4))
    else:
        raise ValueError(mode)
    return pooled


def central_diff_x(x):
    gx = np.zeros_like(x)
    gx[:, :, 1:-1] = 0.5 * (x[:, :, 2:] - x[:, :, :-2])
    gx[:, :, 0] = x[:, :, 1] - x[:, :, 0]
    gx[:, :, -1] = x[:, :, -1] - x[:, :, -2]
    return gx


def central_diff_y(x):
    gy = np.zeros_like(x)
    gy[:, 1:-1, :] = 0.5 * (x[:, 2:, :] - x[:, :-2, :])
    gy[:, 0, :] = x[:, 1, :] - x[:, 0, :]
    gy[:, -1, :] = x[:, -1, :] - x[:, -2, :]
    return gy


def pooled_hog(gray, block_sizes=(4, 8), n_bins=8, eps=1e-8):
    gx = central_diff_x(gray)
    gy = central_diff_y(gray)
    mag = np.sqrt(gx * gx + gy * gy + 1e-12)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    pos = angle * (n_bins / np.pi)
    bin_left = np.floor(pos).astype(int) % n_bins
    frac = pos - np.floor(pos)
    bin_right = (bin_left + 1) % n_bins

    hog_mean_feats = []
    hog_max_feats = []
    grad_max_feats = []
    local_energy_feats = []
    abs_gx = np.abs(gx)
    abs_gy = np.abs(gy)
    energy = gx * gx + gy * gy

    for bs in block_sizes:
        mean_list = []
        max_list = []
        for b in range(n_bins):
            w_left = ((bin_left == b) * (1.0 - frac)).astype(np.float32)
            w_right = ((bin_right == b) * frac).astype(np.float32)
            hist_bin = mag * (w_left + w_right)
            mean_list.append(block_reduce_single_channel(hist_bin, bs, "mean"))
            max_list.append(block_reduce_single_channel(hist_bin, bs, "max"))

        hog_mean_grid = np.stack(mean_list, axis=-1)
        hog_max_grid = np.stack(max_list, axis=-1)
        hog_mean_grid = hog_mean_grid / np.sqrt(np.sum(hog_mean_grid**2, axis=-1, keepdims=True) + eps)
        hog_max_grid = hog_max_grid / np.sqrt(np.sum(hog_max_grid**2, axis=-1, keepdims=True) + eps)
        hog_mean_feats.append(hog_mean_grid.reshape(gray.shape[0], -1))
        hog_max_feats.append(hog_max_grid.reshape(gray.shape[0], -1))

        grad_max = np.stack(
            [
                block_reduce_single_channel(abs_gx, bs, "max"),
                block_reduce_single_channel(abs_gy, bs, "max"),
                block_reduce_single_channel(mag, bs, "max"),
            ],
            axis=-1,
        )
        grad_max_feats.append(grad_max.reshape(gray.shape[0], -1))

        local_energy = np.stack(
            [
                block_reduce_single_channel(energy, bs, "mean"),
                block_reduce_single_channel(energy, bs, "max"),
                block_reduce_single_channel(mag, bs, "std"),
            ],
            axis=-1,
        )
        local_energy_feats.append(local_energy.reshape(gray.shape[0], -1))

    return {
        "hog_mean": np.concatenate(hog_mean_feats, axis=1).astype(np.float32),
        "hog_max": np.concatenate(hog_max_feats, axis=1).astype(np.float32),
        "grad_max": np.concatenate(grad_max_feats, axis=1).astype(np.float32),
        "local_energy": np.concatenate(local_energy_feats, axis=1).astype(np.float32),
    }


def census_texture(gray, block_sizes=(4, 8)):
    c = gray[:, 1:-1, 1:-1]
    n0 = gray[:, :-2, :-2] > c
    n1 = gray[:, :-2, 1:-1] > c
    n2 = gray[:, :-2, 2:] > c
    n3 = gray[:, 1:-1, :-2] > c
    n4 = gray[:, 1:-1, 2:] > c
    n5 = gray[:, 2:, :-2] > c
    n6 = gray[:, 2:, 1:-1] > c
    n7 = gray[:, 2:, 2:] > c
    code = (
        n0.astype(np.uint8)
        + 2 * n1.astype(np.uint8)
        + 4 * n2.astype(np.uint8)
        + 8 * n3.astype(np.uint8)
        + 16 * n4.astype(np.uint8)
        + 32 * n5.astype(np.uint8)
        + 64 * n6.astype(np.uint8)
        + 128 * n7.astype(np.uint8)
    ).astype(np.float32) / 255.0
    padded = np.zeros((gray.shape[0], 32, 32), dtype=np.float32)
    padded[:, 1:-1, 1:-1] = code
    feats = []
    for bs in block_sizes:
        m = block_reduce_single_channel(padded, bs, "mean")
        s = block_reduce_single_channel(padded, bs, "std")
        x = block_reduce_single_channel(padded, bs, "max")
        feats.append(np.stack([m, s, x], axis=-1).reshape(gray.shape[0], -1))
    return np.concatenate(feats, axis=1).astype(np.float32)


def laplace_energy(gray, block_sizes=(4, 8)):
    c = gray
    lap = np.zeros_like(c)
    lap[:, 1:-1, 1:-1] = (
        4.0 * c[:, 1:-1, 1:-1]
        - c[:, :-2, 1:-1]
        - c[:, 2:, 1:-1]
        - c[:, 1:-1, :-2]
        - c[:, 1:-1, 2:]
    )
    ab = np.abs(lap)
    sq = lap * lap
    feats = []
    for bs in block_sizes:
        f = np.stack(
            [
                block_reduce_single_channel(ab, bs, "mean"),
                block_reduce_single_channel(ab, bs, "max"),
                block_reduce_single_channel(sq, bs, "mean"),
                block_reduce_single_channel(sq, bs, "max"),
            ],
            axis=-1,
        )
        feats.append(f.reshape(gray.shape[0], -1))
    return np.concatenate(feats, axis=1).astype(np.float32)


def opp_grad_features(opp, block_sizes=(4, 8), n_bins=6):
    rg = opp[..., 1]
    yb = opp[..., 2]
    g1 = pooled_hog(rg, block_sizes=block_sizes, n_bins=n_bins)
    g2 = pooled_hog(yb, block_sizes=block_sizes, n_bins=n_bins)
    return np.concatenate([g1["hog_mean"], g2["hog_mean"]], axis=1).astype(np.float32)


def orient_energy_features(opp, block_sizes=(4, 8)):
    maps = [opp[..., 0], opp[..., 1], opp[..., 2]]
    thetas = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
    feats = []
    for m in maps:
        gx = central_diff_x(m)
        gy = central_diff_y(m)
        for t in thetas:
            ct = np.float32(np.cos(t))
            st = np.float32(np.sin(t))
            resp = np.abs(ct * gx + st * gy).astype(np.float32)
            for bs in block_sizes:
                mu = block_reduce_single_channel(resp, bs, "mean")
                mx = block_reduce_single_channel(resp, bs, "max")
                sd = block_reduce_single_channel(resp, bs, "std")
                feats.append(np.stack([mu, mx, sd], axis=-1).reshape(resp.shape[0], -1))
    return np.concatenate(feats, axis=1).astype(np.float32)


def build_groups_69(X):
    imgs = vector_to_image_batch(X)
    opp = opponent_channels(imgs)
    gray = opp[..., 0]

    raw_means = []
    raw_stds = []
    opp_means = []
    opp_stds = []
    for bs in (2, 4):
        raw_means.append(block_reduce_from_images(imgs, bs, "mean"))
        raw_stds.append(block_reduce_from_images(imgs, bs, "std"))
        opp_means.append(block_reduce_from_images(opp, bs, "mean"))
        opp_stds.append(block_reduce_from_images(opp, bs, "std"))

    groups = {
        "raw_means": np.concatenate(raw_means, axis=1).astype(np.float32),
        "raw_stds": np.concatenate(raw_stds, axis=1).astype(np.float32),
        "opp_means": np.concatenate(opp_means, axis=1).astype(np.float32),
        "opp_stds": np.concatenate(opp_stds, axis=1).astype(np.float32),
    }
    groups.update(pooled_hog(gray, block_sizes=(4, 8), n_bins=8))
    groups["census"] = census_texture(gray, block_sizes=(4, 8))
    groups["laplace"] = laplace_energy(gray, block_sizes=(4, 8))
    groups["opp_grad"] = opp_grad_features(opp, block_sizes=(4, 8), n_bins=6)
    groups["orient_energy"] = orient_energy_features(opp, block_sizes=(4, 8))
    return groups



class Standardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std_ = X.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def standardize_groups(groups_train, groups_eval):
    scalers = {}
    train_std = {}
    eval_std = {}
    for g in groups_train:
        s = Standardizer()
        train_std[g] = s.fit_transform(groups_train[g])
        eval_std[g] = s.transform(groups_eval[g])
        scalers[g] = s
    return train_std, eval_std, scalers


def apply_standardizers(groups_eval, scalers):
    eval_std = {}
    for g in groups_eval:
        eval_std[g] = scalers[g].transform(groups_eval[g])
    return eval_std


def squared_distances(X1, X2):
    X1 = np.asarray(X1, dtype=np.float32)
    X2 = np.asarray(X2, dtype=np.float32)
    X1_sq = np.sum(X1 * X1, axis=1, keepdims=True, dtype=np.float32)
    X2_sq = np.sum(X2 * X2, axis=1, keepdims=True, dtype=np.float32).T
    D = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
    return np.maximum(D, 0.0).astype(np.float32)


def rbf_kernel_from_distances(D, gamma):
    return np.exp(-gamma * D).astype(np.float32)


def heuristic_gamma(X, subset_size=700, seed=0):
    rng = np.random.default_rng(seed)
    m = min(subset_size, X.shape[0])
    idx = rng.choice(X.shape[0], size=m, replace=False)
    D = squared_distances(X[idx], X[idx])
    mask = ~np.eye(m, dtype=bool)
    median_dist2 = np.median(D[mask])
    return 1.0 / max(median_dist2, 1e-12)


def center_kernel_train(K):
    mean_col = K.mean(axis=0, keepdims=True)
    mean_row = K.mean(axis=1, keepdims=True)
    mean_all = K.mean()
    return (K - mean_row - mean_col + mean_all).astype(np.float32)


def center_kernel_test(K_test, K_train):
    train_col_mean = K_train.mean(axis=0, keepdims=True)
    test_row_mean = K_test.mean(axis=1, keepdims=True)
    train_mean = K_train.mean()
    return (K_test - train_col_mean - test_row_mean + train_mean).astype(np.float32)


def normalize_kernel_pair(K_train, K_test, eps=1e-12):
    diag_mean = np.mean(np.diag(K_train))
    scale = max(diag_mean, eps)
    K_train /= np.float32(scale)
    K_test /= np.float32(scale)
    return K_train, K_test


def build_kernel_pair(train_groups, eval_groups, weights, gamma_mult, base_gamma, dist_train=None, dist_eval=None):
    n_train = next(iter(train_groups.values())).shape[0]
    n_eval = next(iter(eval_groups.values())).shape[0]
    K_train = np.zeros((n_train, n_train), dtype=np.float32)
    K_eval = np.zeros((n_eval, n_train), dtype=np.float32)
    for g in train_groups:
        gamma = base_gamma[g] * gamma_mult.get(g, 1.0)
        if dist_train is None:
            Dt = squared_distances(train_groups[g], train_groups[g])
        else:
            Dt = dist_train[g]
        if dist_eval is None:
            De = squared_distances(eval_groups[g], train_groups[g])
        else:
            De = dist_eval[g]
        Kt_raw = rbf_kernel_from_distances(Dt, gamma)
        Ke_raw = rbf_kernel_from_distances(De, gamma)
        if g in FIXED_MIX_GROUPS:
            g2 = gamma * 2.0
            Kt_raw_2 = rbf_kernel_from_distances(Dt, g2)
            Ke_raw_2 = rbf_kernel_from_distances(De, g2)
            Kt_raw *= np.float32(0.7)
            np.multiply(Kt_raw_2, np.float32(0.3), out=Kt_raw_2)
            Kt_raw += Kt_raw_2
            Ke_raw *= np.float32(0.7)
            np.multiply(Ke_raw_2, np.float32(0.3), out=Ke_raw_2)
            Ke_raw += Ke_raw_2
            del Kt_raw_2, Ke_raw_2
        Kt = center_kernel_train(Kt_raw)
        Ke = center_kernel_test(Ke_raw, Kt_raw)
        Kt, Ke = normalize_kernel_pair(Kt, Ke)
        w = np.float32(weights.get(g, 1.0))
        K_train += w * Kt
        K_eval += w * Ke
        if dist_train is None:
            del Dt
        if dist_eval is None:
            del De
        del Kt_raw, Ke_raw, Kt, Ke
    return K_train, K_eval


class KernelRidgeOVR:
    def __init__(self, reg=1.0, num_classes=10):
        self.reg = float(reg)
        self.num_classes = int(num_classes)
        self.alpha = None

    def fit_from_precomputed_kernel(self, K_train, y):
        n = K_train.shape[0]
        A = K_train.astype(np.float64) + self.reg * np.eye(n, dtype=np.float64)
        Y = -np.ones((n, self.num_classes), dtype=np.float64)
        for c in range(self.num_classes):
            Y[y == c, c] = 1.0
        self.alpha = np.linalg.solve(A, Y)

    def score(self, K_test):
        return K_test.astype(np.float64) @ self.alpha



def estimate_cache_bytes(n_train, n_eval, n_groups):
    bytes_per = 4
    total = n_groups * (n_train * n_train + 2 * n_eval * n_train) * bytes_per
    return total


def maybe_precompute_distances(train_groups, eval_groups_orig, eval_groups_flip):
    if not CACHE_DISTANCES:
        return None
    n_train = next(iter(train_groups.values())).shape[0]
    n_eval = next(iter(eval_groups_orig.values())).shape[0]
    n_groups = len(train_groups)
    total_bytes = estimate_cache_bytes(n_train, n_eval, n_groups)
    if total_bytes > MAX_CACHE_GB * (1024**3):
        print("Skipping distance cache (estimated size too large).")
        return None
    print("Precomputing distance cache...")
    dist_train = {}
    dist_eval = {}
    dist_eval_flip = {}
    for g in train_groups:
        dist_train[g] = squared_distances(train_groups[g], train_groups[g])
        dist_eval[g] = squared_distances(eval_groups_orig[g], train_groups[g])
        dist_eval_flip[g] = squared_distances(eval_groups_flip[g], train_groups[g])
    return {"train": dist_train, "eval": dist_eval, "eval_flip": dist_eval_flip}


def evaluate_config(
    train_groups,
    eval_groups_orig,
    eval_groups_flip,
    y_train,
    y_val,
    weights,
    gamma_mult,
    reg,
    base_gamma,
    dist_cache=None,
):
    dist_train = None
    dist_eval = None
    dist_eval_flip = None
    if dist_cache is not None:
        dist_train = dist_cache["train"]
        dist_eval = dist_cache["eval"]
        dist_eval_flip = dist_cache["eval_flip"]

    Kt, Keo = build_kernel_pair(
        train_groups, eval_groups_orig, weights, gamma_mult, base_gamma, dist_train=dist_train, dist_eval=dist_eval
    )
    _, Keh = build_kernel_pair(
        train_groups, eval_groups_flip, weights, gamma_mult, base_gamma, dist_train=dist_train, dist_eval=dist_eval_flip
    )
    clf = KernelRidgeOVR(reg=reg, num_classes=NUM_CLASSES)
    clf.fit_from_precomputed_kernel(Kt, y_train)
    So = clf.score(Keo)
    Sh = clf.score(Keh)
    S = np.empty_like(So, dtype=np.float64)
    for c in range(NUM_CLASSES):
        a = float(CLASS_ALPHA[c])
        S[:, c] = a * So[:, c] + (1.0 - a) * Sh[:, c]
    pred = np.argmax(S, axis=1)
    acc = float(np.mean(pred == y_val))
    return acc


def tune_hyperparameters(Xtr, ytr, Xval, yval):
    Xval_flip = transform_vectors(Xval, "hflip")

    tr_groups = build_groups_69(Xtr)
    val_groups = build_groups_69(Xval)
    val_groups_flip = build_groups_69(Xval_flip)

    tr_std, val_std, scalers = standardize_groups(tr_groups, val_groups)
    val_flip_std = apply_standardizers(val_groups_flip, scalers)

    base_gamma = {k: heuristic_gamma(tr_std[k], subset_size=GAMMA_SUBSET, seed=0) for k in tr_std}
    group_list = list(tr_std.keys())

    init_weights = {g: float(BASE_WEIGHTS.get(g, 1.0)) for g in group_list}
    init_gamma = {g: float(BASE_GAMMA_MULT.get(g, 1.0)) for g in group_list}
    init_reg = float(BASE_REG)

    dist_cache = maybe_precompute_distances(tr_std, val_std, val_flip_std)

    best_weights = dict(init_weights)
    best_gamma = dict(init_gamma)
    best_reg = init_reg
    best_acc = evaluate_config(
        tr_std,
        val_std,
        val_flip_std,
        ytr,
        yval,
        best_weights,
        best_gamma,
        best_reg,
        base_gamma,
        dist_cache=dist_cache,
    )
    print(f"Baseline val accuracy: {100*best_acc:.2f}%")

    print(f"Baseline val accuracy: {100*best_acc:.2f}%")

    for reg in REG_GRID:
        acc = evaluate_config(
            tr_std, val_std, val_flip_std,
            ytr, yval,
            best_weights, best_gamma, reg,
            base_gamma,
            dist_cache=dist_cache,
        )
        print(f"[reg] reg={reg}: acc={100*acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_reg = float(reg)
            print("  new best reg")

    rng = np.random.default_rng(RANDOM_SEED)
    for t in range(N_TRIALS_WEIGHTS):
        trial_weights = dict(best_weights)
        for g in group_list:
            trial_weights[g] = float(rng.choice(WEIGHT_GRID))

        acc = evaluate_config(
            tr_std, val_std, val_flip_std,
            ytr, yval,
            trial_weights, best_gamma, best_reg,
            base_gamma,
            dist_cache=dist_cache,
        )
        print(f"[weights] trial {t+1}/{N_TRIALS_WEIGHTS}: acc={100*acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_weights = dict(trial_weights)
            print("  new best weights")

    for t in range(N_TRIALS_GAMMA):
        trial_gamma = dict(best_gamma)
        for g in group_list:
            trial_gamma[g] = float(rng.choice(GAMMA_GRID))

        acc = evaluate_config(
            tr_std, val_std, val_flip_std,
            ytr, yval,
            best_weights, trial_gamma, best_reg,
            base_gamma,
            dist_cache=dist_cache,
        )
        print(f"[gamma] trial {t+1}/{N_TRIALS_GAMMA}: acc={100*acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_gamma = dict(trial_gamma)
            print("  new best gamma")
            
    for coord_pass in range(COORD_PASSES):
        print(f"\n=== Coordinate pass {coord_pass+1}/{COORD_PASSES} ===")
        improved = False

        local_best_acc = best_acc
        local_best_reg = best_reg
        for reg in REG_GRID:
            acc = evaluate_config(
                tr_std, val_std, val_flip_std,
                ytr, yval,
                best_weights, best_gamma, reg,
                base_gamma,
                dist_cache=dist_cache,
            )
            print(f"[coord-reg] reg={reg}: acc={100*acc:.2f}%")
            if acc > local_best_acc:
                local_best_acc = acc
                local_best_reg = float(reg)

        if local_best_acc > best_acc:
            best_acc = local_best_acc
            best_reg = local_best_reg
            improved = True
            print(f"  accepted reg={best_reg}")

        for g in group_list:
            local_best_acc = best_acc
            local_best_val = best_weights[g]

            for w in WEIGHT_GRID:
                trial_weights = dict(best_weights)
                trial_weights[g] = float(w)

                acc = evaluate_config(
                    tr_std, val_std, val_flip_std,
                    ytr, yval,
                    trial_weights, best_gamma, best_reg,
                    base_gamma,
                    dist_cache=dist_cache,
                )
                print(f"[coord-weight] {g}={w}: acc={100*acc:.2f}%")

                if acc > local_best_acc:
                    local_best_acc = acc
                    local_best_val = float(w)

            if local_best_acc > best_acc:
                best_acc = local_best_acc
                best_weights[g] = local_best_val
                improved = True
                print(f"  accepted weight[{g}]={local_best_val}")

        for g in group_list:
            local_best_acc = best_acc
            local_best_val = best_gamma[g]

            for gm in GAMMA_GRID:
                trial_gamma = dict(best_gamma)
                trial_gamma[g] = float(gm)

                acc = evaluate_config(
                    tr_std, val_std, val_flip_std,
                    ytr, yval,
                    best_weights, trial_gamma, best_reg,
                    base_gamma,
                    dist_cache=dist_cache,
                )
                print(f"[coord-gamma] {g}={gm}: acc={100*acc:.2f}%")

                if acc > local_best_acc:
                    local_best_acc = acc
                    local_best_val = float(gm)

            if local_best_acc > best_acc:
                best_acc = local_best_acc
                best_gamma[g] = local_best_val
                improved = True
                print(f"  accepted gamma[{g}]={local_best_val}")

        if not improved:
            print("No improvement on this coordinate pass, stopping early.")
            break

    return best_weights, best_gamma, best_reg, best_acc


def save_params(path, weights, gamma_mult, reg, acc):
    payload = {
        "weights": {k: float(v) for k, v in weights.items()},
        "gamma_mult": {k: float(v) for k, v in gamma_mult.items()},
        "reg": float(reg),
        "val_acc": float(acc),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["weights"], payload["gamma_mult"], float(payload["reg"])


def compute_scores_with_params(Xtr, ytr, Xe_orig, Xe_flip, weights, gamma_mult, reg):
    tr_groups = build_groups_69(Xtr)
    eval_groups = build_groups_69(Xe_orig)
    eval_flip = build_groups_69(Xe_flip)

    tr_std, eval_std, scalers = standardize_groups(tr_groups, eval_groups)
    eval_flip_std = apply_standardizers(eval_flip, scalers)

    base_gamma = {k: heuristic_gamma(tr_std[k], subset_size=GAMMA_SUBSET, seed=0) for k in tr_std}
    Kt, Keo = build_kernel_pair(tr_std, eval_std, weights, gamma_mult, base_gamma)
    _, Keh = build_kernel_pair(tr_std, eval_flip_std, weights, gamma_mult, base_gamma)

    clf = KernelRidgeOVR(reg=reg, num_classes=NUM_CLASSES)
    clf.fit_from_precomputed_kernel(Kt, ytr)
    So = clf.score(Keo)
    Sh = clf.score(Keh)

    S = np.empty_like(So, dtype=np.float64)
    for c in range(NUM_CLASSES):
        a = float(CLASS_ALPHA[c])
        S[:, c] = a * So[:, c] + (1.0 - a) * Sh[:, c]
    return S



def run_validate(params_path):
    X = load_X("Xtr.csv")
    y = load_y("Ytr.csv")
    Xtr, ytr, Xv, yv = stratified_train_val_split(X, y, val_ratio=VAL_RATIO, seed=SPLIT_SEED)

    best_weights, best_gamma, best_reg, best_acc = tune_hyperparameters(Xtr, ytr, Xv, yv)
    print("\nBest validation accuracy: {:.2f}%".format(100 * best_acc))
    print("Best reg:", best_reg)
    print("Best weights:")
    for k in sorted(best_weights):
        print(f"  {k}: {best_weights[k]:.3f}")
    print("Best gamma_mult:")
    for k in sorted(best_gamma):
        print(f"  {k}: {best_gamma[k]:.3f}")

    save_params(params_path, best_weights, best_gamma, best_reg, best_acc)
    print(f"Saved tuned params to {params_path}")


def run_submit(params_path, output_csv):
    if os.path.exists(params_path):
        weights, gamma_mult, reg = load_params(params_path)
        print(f"Loaded tuned params from {params_path}")
    else:
        print("Params file not found, tuning on a validation split first...")
        X = load_X("Xtr.csv")
        y = load_y("Ytr.csv")
        Xtr, ytr, Xv, yv = stratified_train_val_split(X, y, val_ratio=VAL_RATIO, seed=SPLIT_SEED)
        weights, gamma_mult, reg, _ = tune_hyperparameters(Xtr, ytr, Xv, yv)
        save_params(params_path, weights, gamma_mult, reg, 0.0)

    Xtr = load_X("Xtr.csv")
    ytr = load_y("Ytr.csv")
    Xte = load_X("Xte.csv")
    Xteh = transform_vectors(Xte, "hflip")
    S = compute_scores_with_params(Xtr, ytr, Xte, Xteh, weights, gamma_mult, reg)
    pred = np.argmax(S, axis=1).astype(np.int64)
    sub = pd.DataFrame({"Id": np.arange(1, Xte.shape[0] + 1), "Prediction": pred})
    sub.to_csv(output_csv, index=False)
    print(f"Saved submission: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "submit", "both"], default="validate")
    parser.add_argument("--output", type=str, default="submission_69p2_tuned.csv")
    parser.add_argument("--params", type=str, default=PARAMS_PATH)
    args = parser.parse_args()

    if args.mode == "validate":
        run_validate(args.params)
    elif args.mode == "submit":
        run_submit(args.params, args.output)
    else:
        run_validate(args.params)
        run_submit(args.params, args.output)


if __name__ == "__main__":
    main()
