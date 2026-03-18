"""
start.py — Train three CKN3 variants and generate Y_pred.csv

Pipeline
--------
1. Load & normalise data (train/val split + test set)
2. Phase 1  — Unsupervised CKN layer training (shared by all models)
              → weights saved to weights.pkl
3. Phase 2a — CKN3 + LinearSVM              → Y_pred_ckn.csv
4. Phase 2b — CKN3 + HOG + LinearSVM        → Y_pred_ckn_hog.csv
              (CKN weights reused from Phase 1, no retraining)
5. Phase 2c — CKN3 + Crammer-Singer SVM     → Y_pred_ckn_cs.csv
              (CKN weights reused from Phase 1, no retraining)
6. Pick the model with the highest dev accuracy → Y_pred.csv

Usage
-----
    python start.py
"""

import numpy as np
import pandas as pd
import pickle
from timeit import default_timer as timer

from data import DataLoader
from model import CKN3
from crammer_singer import CrammerSingerSVMClassifier

# ── Hyperparameters ────────────────────────────────────────────────────────────
DATA_PATH          = './data/'
BATCH_SIZE         = 256
VAL_RATIO          = 0.2

# Shared CKN + LinearSVM settings
ALPHA              = 0.001
N_SAMPLING_PATCHES = 300000
MAXITER_BFGS       = 5000

# HOG options (Phase 2b)
HOG_CELL_SIZE      = (4, 4)
HOG_BLOCK_SIZE     = (2, 2)
HOG_BLOCK_STRIDE   = (1, 1)
HOG_N_BINS         = 9

# Crammer-Singer options (Phase 2c)
CS_KERNEL          = 'rbf'
CS_GAMMA           = 0.1
CS_ALPHA           = 0.001   # same regularisation convention as LinearSVM
CS_MAXITER         = 10_000
# ──────────────────────────────────────────────────────────────────────────────


def load_data():
    print("=" * 55)
    print("Loading data")
    print("=" * 55)
    Xtr_raw = np.array(pd.read_csv(f'{DATA_PATH}Xtr.csv', header=None, sep=',',
                                    usecols=range(3072), encoding='unicode_escape'))
    Ytr_raw = np.array(pd.read_csv(f'{DATA_PATH}Ytr.csv', sep=',', usecols=[1])).squeeze()
    Xte_raw = np.array(pd.read_csv(f'{DATA_PATH}Xte.csv', header=None, sep=',',
                                    usecols=range(3072), encoding='unicode_escape'))

    Xtr_raw = Xtr_raw.reshape(-1, 3, 32, 32).astype(np.float32)
    Xte_raw = Xte_raw.reshape(-1, 3, 32, 32).astype(np.float32)

    # Normalise by channel (stats from training set only — no leakage to test)
    channel_mean = Xtr_raw.mean(axis=(0, 2, 3), keepdims=True)
    channel_std  = Xtr_raw.std(axis=(0, 2, 3),  keepdims=True)
    Xtr_raw = (Xtr_raw - channel_mean) / (channel_std + 1e-6)
    Xte_raw = (Xte_raw - channel_mean) / (channel_std + 1e-6)

    # Train / validation split
    n     = len(Xtr_raw)
    n_val = int(n * VAL_RATIO)
    idx   = np.random.permutation(n)

    X_dev,   Y_dev   = Xtr_raw[idx[:n_val]],  Ytr_raw[idx[:n_val]]
    X_train, Y_train = Xtr_raw[idx[n_val:]],  Ytr_raw[idx[n_val:]]

    print(f"X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"X_dev:   {X_dev.shape}    Y_dev:   {Y_dev.shape}")
    print(f"X_test:  {Xte_raw.shape}")
    print(f"Normalisation — mean: {X_train.mean():.4f}, std: {X_train.std():.4f}\n")

    return X_train, Y_train, X_dev, Y_dev, Xte_raw


def evaluate(model, loader, split_name):
    preds, targets = [], []
    for X_batch, y_batch in loader:
        preds.append(np.argmax(model(X_batch), axis=1))
        targets.append(y_batch)
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    acc = np.mean(preds == targets)
    print(f"  {split_name:6s} accuracy: {acc * 100:.2f}%")
    return acc


def predict(model, X, batch_size=256):
    """Run inference on X and return class predictions."""
    loader = DataLoader(X, y=None, batch_size=batch_size, shuffle=False)
    preds = []
    for X_batch in loader:
        preds.append(np.argmax(model(X_batch), axis=1))
    return np.concatenate(preds)


def save_predictions(preds, path):
    df = pd.DataFrame({'Id': np.arange(1, len(preds) + 1), 'Prediction': preds})
    df.to_csv(path, index=False)
    print(f"  Saved {len(preds)} predictions → {path}")


def main():
    # ── 1. Data ────────────────────────────────────────────────────────────────
    X_train, Y_train, X_dev, Y_dev, X_test = load_data()

    train_loader = DataLoader(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(X_dev,   Y_dev,   batch_size=BATCH_SIZE, shuffle=False)

    # ── 2. Phase 1 — Unsupervised CKN training (shared by all models) ──────────
    # All three variants use the identical CKN3 feature extractor, so we train
    # the convolutional layers once and reuse the weights for phases 2b and 2c.
    print("=" * 55)
    print("Phase 1 — Unsupervised CKN layer training (shared)")
    print("=" * 55)

    base_model = CKN3(alpha=ALPHA, maxiter=MAXITER_BFGS)
    base_model.unsup_train_ckn(train_loader, N_SAMPLING_PATCHES)

    trained_weights = [layer.weight.values.copy()
                       for layer in base_model.features.ckn_layers]
    with open('weights.pkl', 'wb') as f:
        pickle.dump(trained_weights, f)
    print("\nCKN weights saved to weights.pkl")

    results = {}   # label -> (train_acc, dev_acc, test_preds)

    # ── 3. Phase 2a — CKN3 + LinearSVM ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Phase 2a — CKN3 + LinearSVM")
    print("=" * 55)

    tic = timer()
    base_model.unsup_train_classifier(train_loader)
    print(f"  L-BFGS finished in {timer() - tic:.1f}s")

    tr = evaluate(base_model, train_loader, "Train")
    dv = evaluate(base_model, dev_loader,   "Dev")
    preds = predict(base_model, X_test, BATCH_SIZE)
    save_predictions(preds, 'Y_pred_ckn.csv')
    results['CKN3 + LinearSVM'] = (tr, dv, preds)

    # ── 4. Phase 2b — CKN3 + HOG + LinearSVM ──────────────────────────────────
    print("\n" + "=" * 55)
    print("Phase 2b — CKN3 + HOG + LinearSVM  (CKN weights reused)")
    print("=" * 55)

    hog_model = CKN3(
        alpha=ALPHA, maxiter=MAXITER_BFGS,
        use_hog=True,
        hog_cell_size=HOG_CELL_SIZE,
        hog_block_size=HOG_BLOCK_SIZE,
        hog_block_stride=HOG_BLOCK_STRIDE,
        hog_n_bins=HOG_N_BINS,
    )
    print(f"  CKN features:   {hog_model.ckn_out_features}")
    print(f"  HOG features:   {hog_model.hog_out_features}")
    print(f"  Total features: {hog_model.out_features}")

    hog_model.load_ckn_weights(trained_weights)

    tic = timer()
    hog_model.unsup_train_classifier(train_loader)
    print(f"  L-BFGS finished in {timer() - tic:.1f}s")

    tr = evaluate(hog_model, train_loader, "Train")
    dv = evaluate(hog_model, dev_loader,   "Dev")
    preds = predict(hog_model, X_test, BATCH_SIZE)
    save_predictions(preds, 'Y_pred_ckn_hog.csv')
    results['CKN3 + HOG + LinearSVM'] = (tr, dv, preds)

    # ── 5. Phase 2c — CKN3 + Crammer-Singer SVM ───────────────────────────────
    print("\n" + "=" * 55)
    print("Phase 2c — CKN3 + Crammer-Singer SVM  (CKN weights reused)")
    print("=" * 55)

    cs_model = CKN3(
        alpha=CS_ALPHA,
        maxiter=CS_MAXITER,
        classifier_cls=CrammerSingerSVMClassifier,
        classifier_kwargs={
            'kernel': CS_KERNEL,
            'gamma':  CS_GAMMA,
        },
    )
    print(f"  Kernel: {CS_KERNEL}, gamma: {CS_GAMMA}")

    cs_model.load_ckn_weights(trained_weights)

    tic = timer()
    cs_model.unsup_train_classifier(train_loader)
    print(f"  Crammer-Singer finished in {timer() - tic:.1f}s")
    if hasattr(cs_model.classifier, 'n_support_'):
        print(f"  Support patterns: {cs_model.classifier.n_support_}")

    tr = evaluate(cs_model, train_loader, "Train")
    dv = evaluate(cs_model, dev_loader,   "Dev")
    preds = predict(cs_model, X_test, BATCH_SIZE)
    save_predictions(preds, 'Y_pred_ckn_cs.csv')
    results['CKN3 + Crammer-Singer'] = (tr, dv, preds)

    # ── 6. Summary & best submission ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Summary")
    print("=" * 55)
    print(f"  {'Model':<28} {'Train acc':>10} {'Dev acc':>10}")
    print(f"  {'-' * 50}")

    best_label, best_dv, best_preds = None, -1.0, None
    for label, (tr, dv, preds) in results.items():
        marker = " ◀" if dv > best_dv else ""
        print(f"  {label:<28} {tr*100:>9.2f}% {dv*100:>9.2f}%{marker}")
        if dv > best_dv:
            best_dv, best_label, best_preds = dv, label, preds

    print(f"\n  Best model: {best_label}  (dev acc {best_dv*100:.2f}%)")
    save_predictions(best_preds, 'Y_pred.csv')
    print("\nDone.")


if __name__ == '__main__':
    main()
