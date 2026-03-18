import argparse
import json
import numpy as np
import pandas as pd


def load_X(path):
    chunks = []
    for ch in pd.read_csv(path, header=None, usecols=range(3072), chunksize=500):
        chunks.append(ch.values.astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def load_y(path):
    return np.array(pd.read_csv(path, usecols=[1])).squeeze().astype(np.int64)


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


def build_groups(X, block_sizes_stats=(2, 4), block_sizes_hog=(4, 8), n_bins=8, add_extra=True):
    imgs = vector_to_image_batch(X)
    opp = opponent_channels(imgs)
    gray = opp[..., 0]

    raw_means = []
    raw_stds = []
    opp_means = []
    opp_stds = []
    for bs in block_sizes_stats:
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
    groups.update(pooled_hog(gray, block_sizes=block_sizes_hog, n_bins=n_bins))
    if add_extra:
        groups["census"] = census_texture(gray, block_sizes=(4, 8))
        groups["laplace"] = laplace_energy(gray, block_sizes=(4, 8))
        groups["opp_grad"] = opp_grad_features(opp, block_sizes=(4, 8), n_bins=6)
    return groups


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
    groups = build_groups(X, block_sizes_stats=(2, 4), block_sizes_hog=(4, 8), n_bins=8, add_extra=True)
    imgs = vector_to_image_batch(X)
    opp = opponent_channels(imgs)
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


def build_kernel_pair(train_groups, eval_groups, weights, gamma_mult, base_gamma):
    n_train = next(iter(train_groups.values())).shape[0]
    n_eval = next(iter(eval_groups.values())).shape[0]
    K_train = np.zeros((n_train, n_train), dtype=np.float32)
    K_eval = np.zeros((n_eval, n_train), dtype=np.float32)

    for g in train_groups:
        gamma = base_gamma[g] * gamma_mult.get(g, 1.0)
        Dt = squared_distances(train_groups[g], train_groups[g])
        De = squared_distances(eval_groups[g], train_groups[g])

        Kt_raw = rbf_kernel_from_distances(Dt, gamma)
        Ke_raw = rbf_kernel_from_distances(De, gamma)

        if g in {"hog_mean", "hog_max", "opp_grad"}:
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

        del Dt, De, Kt_raw, Ke_raw, Kt, Ke

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


BASE = {
    "block_sizes_stats": (2, 4),
    "block_sizes_hog": (4, 8),
    "n_bins": 8,
    "add_extra": True,
}


def load_tuned_params(json_path="tuned_params_69p2.json"):
    with open(json_path, "r") as f:
        params = json.load(f)

    weights = {k: float(v) for k, v in params["weights"].items()}
    gamma_mult = {k: float(v) for k, v in params["gamma_mult"].items()}
    reg = float(params["reg"])

    class_alpha = None
    if "class_alpha" in params:
        class_alpha = np.array(params["class_alpha"], dtype=np.float64)

    class_model = None
    if "class_model" in params:
        class_model = np.array(params["class_model"], dtype=np.int64)

    return weights, gamma_mult, reg, class_alpha, class_model


def make_cfg_from_tuned_params(weights, gamma_mult, reg):
    return {
        **BASE,
        "weights": weights,
        "gamma_mult": gamma_mult,
        "reg": reg,
        "alpha_flip": np.array([0.50] * 10, dtype=np.float64),
    }


def fit_model_scores_decomposed(Xtr, ytr, Xe_orig, Xe_hflip, cfg):
    Gtr = build_groups_69(Xtr)
    Geo = build_groups_69(Xe_orig)
    Geh = build_groups_69(Xe_hflip)

    for k in Gtr:
        s = Standardizer()
        Gtr[k] = s.fit_transform(Gtr[k])
        Geo[k] = s.transform(Geo[k])
        Geh[k] = s.transform(Geh[k])

    base_gamma = {k: heuristic_gamma(Gtr[k], subset_size=700, seed=0) for k in Gtr}

    Kt, Keo = build_kernel_pair(Gtr, Geo, cfg["weights"], cfg["gamma_mult"], base_gamma)
    _, Keh = build_kernel_pair(Gtr, Geh, cfg["weights"], cfg["gamma_mult"], base_gamma)

    clf = KernelRidgeOVR(reg=cfg["reg"], num_classes=10)
    clf.fit_from_precomputed_kernel(Kt, ytr)

    So = clf.score(Keo)
    Sh = clf.score(Keh)
    return So, Sh


def run_submit_tuned_from_json(output_csv, json_path="tuned_params_69p2.json"):
    print(f"Loading tuned params from: {json_path}")
    weights, gamma_mult, reg, class_alpha, class_model = load_tuned_params(json_path)
    cfg = make_cfg_from_tuned_params(weights, gamma_mult, reg)

    print("Loading full training data...")
    Xtr = load_X("Xtr.csv")
    ytr = load_y("Ytr.csv")

    print("Loading test data...")
    Xte = load_X("Xte.csv")
    Xteh = transform_vectors(Xte, "hflip")

    print("Training on full Xtr and predicting on Xte...")
    So, Sh = fit_model_scores_decomposed(Xtr, ytr, Xte, Xteh, cfg)

    if class_alpha is not None:
        if class_alpha.shape[0] != So.shape[1]:
            raise ValueError("class_alpha has incompatible length with number of classes.")
        S = np.empty_like(So, dtype=np.float64)
        for c in range(So.shape[1]):
            a = class_alpha[c]
            S[:, c] = a * So[:, c] + (1.0 - a) * Sh[:, c]
    else:
        S = 0.5 * So + 0.5 * Sh

    pred = np.argmax(S, axis=1).astype(np.int64)

    sub = pd.DataFrame({
        "Id": np.arange(1, Xte.shape[0] + 1),
        "Prediction": pred,
    })
    sub.to_csv(output_csv, index=False)
    print(f"Saved submission: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["submit_tuned_json"],
        default="submit_tuned_json",
    )
    parser.add_argument("--output", type=str, default="submission_tuned_69p2.csv")
    parser.add_argument("--json", type=str, default="tuned_params_69p2.json")
    args = parser.parse_args()

    if args.mode == "submit_tuned_json":
        run_submit_tuned_from_json(args.output, args.json)


if __name__ == "__main__":
    main()
