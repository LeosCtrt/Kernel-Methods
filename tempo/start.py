import argparse
import time
import numpy as np
import pandas as pd

from model import CKNet, SUPMODELS
from loss import LOSS
from optimizers import SGD, PatienceLR
from utils import ZCA, accuracy_score, countX, count_parameters

# ─────────────────────────────────────────────────────────────────────────────
#  Path configuration  (edit these for your environment)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/competitions/data-challenge-kernel-methods-2025-2026/"
OUTPUT_CSV = "Yte_pred.csv"

# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def create_dataset(data_path: str = DATA_DIR):
    """
    Load competition CSV files and return numpy arrays.

    Xtr / Xte rows are flat RGB pixels:
        [R_0…R_1023, G_0…G_1023, B_0…B_1023]
    reshaped to (N, 3, 32, 32).

    Returns:
        Xtr (N_tr, 3, 32, 32), Ytr (N_tr,), Xte (N_te, 3, 32, 32)
    """
    print("  Reading Xtr.csv …", flush=True)
    Xtr = np.array(
        pd.read_csv(f"{data_path}Xtr.csv", header=None, sep=",",
                    usecols=range(3072)),
        dtype=np.float32).reshape(-1, 3, 32, 32)

    print("  Reading Xte.csv …", flush=True)
    Xte = np.array(
        pd.read_csv(f"{data_path}Xte.csv", header=None, sep=",",
                    usecols=range(3072)),
        dtype=np.float32).reshape(-1, 3, 32, 32)

    print("  Reading Ytr.csv …", flush=True)
    Ytr = np.array(
        pd.read_csv(f"{data_path}Ytr.csv", sep=",", usecols=[1]),
        dtype=np.int32).squeeze()

    # Normalise to [0, 1] if raw uint8 values
    if Xtr.max() > 1.5:
        Xtr /= 255.0
        Xte /= 255.0

    print(f"  Xtr: {Xtr.shape}  Xte: {Xte.shape}  Ytr: {Ytr.shape}", flush=True)
    return Xtr, Ytr, Xte


# ─────────────────────────────────────────────────────────────────────────────
#  Data loader  (generator-based, matching the other codebase)
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloader_generator(X: np.ndarray, y: np.ndarray = None,
                                batch_size: int = 128,
                                shuffle: bool = True):
    """
    Yield mini-batches of (X_batch, y_batch) or X_batch if y is None.
    Mirrors create_dataloader_generator from start.py of the other codebase.
    """
    N       = len(X)
    indices = np.random.permutation(N) if shuffle else np.arange(N)
    for s in range(0, N, batch_size):
        idx = indices[s:s + batch_size]
        if y is not None:
            yield X[idx], y[idx]
        else:
            yield X[idx]


class NumpyDataLoader:
    """
    Reusable data loader wrapping create_dataloader_generator.
    Mirrors NumpyDataLoader from the other codebase.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray = None,
                 batch_size: int = 128, shuffle: bool = True):
        self.X          = X
        self.y          = y
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.num_samples = len(X)

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        return create_dataloader_generator(
            self.X, self.y, self.batch_size, self.shuffle)


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def sup_train(model: CKNet, data_loaders: dict, args) -> float:
    """
    Supervised training of the full CKNet (CKN filters + classifier head).

    Training alternates between:
      Step A — update classifier W  via mini-batch SGD  (fast, vectorised).
      Step B — update CKN filters Z via backprop        (image-by-image).

    Args:
        model:        Initialised CKNet.
        data_loaders: Dict with keys 'train', 'val', 'init'.
        args:         Parsed command-line / config arguments.

    Returns:
        Best validation accuracy achieved.
    """
    criterion  = LOSS[args.loss]()
    lam_eff    = args.alpha / len(data_loaders["train"].X)

    # ── optimiser for classifier head ────────────────────────────
    W_optim    = SGD(
        [{"params": model.classifier.parameters(),
          "lr": args.lr, "weight_decay": args.alpha,
          "momentum": 0.9}],
        grad_clip=10.0)
    lr_sched   = PatienceLR(W_optim, patience=3, factor=0.5,
                            min_lr=args.lr * 1e-4)

    # ── unsupervised K-means initialisation ──────────────────────
    print("\n[Init] Spherical K-means …", flush=True)
    model.unsup_train_ckn(data_loaders["init"], args.sampling_patches)

    # ── pre-encode training + validation features ─────────────────
    print("\n[Encode] Initial feature encoding …", flush=True)
    Phi_tr, feat_scale = _encode_dataset(model, data_loaders["train"])
    Phi_val, _         = _encode_dataset(model, data_loaders["val"],
                                         feat_scale=feat_scale)
    y_tr  = data_loaders["train"].y
    y_val = data_loaders["val"].y
    C_y   = model.nclass
    D     = Phi_tr.shape[1]

    best_acc    = 0.0
    best_W      = model.classifier.weight.values.copy()
    best_b      = (model.classifier.bias.values.copy()
                   if model.classifier.bias is not None else None)
    best_scale  = feat_scale

    for epoch in range(args.epochs):
        t0 = time.time()

        # ════════════════════════════════════════════════════════
        # STEP A – update classifier W  (mini-batch SGD, convex)
        # ════════════════════════════════════════════════════════
        total_loss  = 0.0
        perm        = np.random.permutation(len(Phi_tr))

        for s in range(0, len(Phi_tr), args.batch_size):
            idx = perm[s:s + args.batch_size]
            Pb  = Phi_tr[idx]; yb = y_tr[idx]
            bsz = len(idx)

            sc = np.clip(model.classifier(Pb), -100.0, 100.0)
            Y  = -np.ones((bsz, C_y))
            Y[np.arange(bsz), yb.astype(int)] = 1.0
            margin = np.maximum(0.0, 1.0 - Y * sc)
            total_loss += (margin ** 2).sum() / (len(Phi_tr) * C_y)

            # Gradients
            dout = -2.0 * Y * margin / (bsz * C_y)
            model.classifier.weight.gradients = dout.T @ Pb \
                                                + lam_eff * model.classifier.weight.values
            if model.classifier.bias is not None:
                model.classifier.bias.gradients = dout.sum(0).reshape(-1, 1)

            W_optim.step()
            W_optim.zero_grad()

        lr_sched.step(total_loss)   # patience-based decay

        # ════════════════════════════════════════════════════════
        # STEP B – update CKN filters Z  (backprop, paper Prop. 1)
        # ════════════════════════════════════════════════════════
        W   = model.classifier.weight.values
        b   = (model.classifier.bias.values.ravel()
               if model.classifier.bias is not None else None)
        z_data   = data_loaders["train"]
        z_idx    = np.random.choice(len(z_data.X),
                                    min(len(z_data.X), args.n_z_samples),
                                    replace=False)
        grad_Z   = [np.zeros_like(l.Z) for l in model.features.ckn_layers]
        n_used   = len(z_idx)

        for img_i in z_idx:
            img   = z_data.X[img_i]
            label = int(z_data.y[img_i])

            I_k, caches = model.features.forward_single(img)
            feat        = I_k.ravel().astype(np.float64) / feat_scale
            if not np.isfinite(feat).all():
                continue

            # Loss gradient w.r.t. features
            scores = W @ feat + (b if b is not None else 0.0)
            Y_1    = np.full(C_y, -1.0); Y_1[label] = 1.0
            margin = np.maximum(0.0, 1.0 - Y_1 * scores)
            ds     = -2.0 * Y_1 * margin / (n_used * C_y)

            # Back-project through classifier
            dfeat  = (W.T @ ds / feat_scale).reshape(I_k.shape).astype(np.float32)
            dfeat  = np.nan_to_num(dfeat, nan=0.0, posinf=0.0, neginf=0.0)
            fnorm  = np.linalg.norm(dfeat)
            if fnorm > 1.0:
                dfeat /= fnorm
            dI = dfeat

            # Backprop through CKN layers
            K = model.features.n_layers
            for k in range(K - 1, -1, -1):
                dZk, dI = model.features.ckn_layers[k].backward(dI, caches[k])
                dZk = np.nan_to_num(dZk, nan=0.0, posinf=0.0, neginf=0.0)
                dz_norm = np.linalg.norm(dZk)
                if dz_norm > 10.0:
                    dZk *= 10.0 / dz_norm
                grad_Z[k] += dZk
                di_norm = np.linalg.norm(dI)
                if di_norm > 10.0:
                    dI = (dI * (10.0 / di_norm)).astype(np.float32)

        # Apply accumulated Z gradients
        for k, layer in enumerate(model.features.ckn_layers):
            layer.step_Z(grad_Z[k] / n_used, lr=args.lr_z, momentum=0.9)

        # Evaluate
        tr_acc  = (model.classifier(Phi_tr ).argmax(1) == y_tr ).mean()
        val_acc = (model.classifier(Phi_val).argmax(1) == y_val).mean()
        lr_now  = W_optim.param_groups[0].get("lr", args.lr)

        print(f"Epoch {epoch + 1:3d}/{args.epochs}  "
              f"loss={total_loss:.4f}  "
              f"tr={100*tr_acc:.1f}%  val={100*val_acc:.1f}%  "
              f"lr={lr_now:.4f}  t={time.time()-t0:.0f}s", flush=True)

        if val_acc > best_acc:
            best_acc   = val_acc
            best_W     = model.classifier.weight.values.copy()
            best_b     = (model.classifier.bias.values.copy()
                          if model.classifier.bias is not None else None)
            best_scale = feat_scale

        # Re-encode features every `feat_recompute` epochs
        if (epoch + 1) % args.feat_recompute == 0 and epoch < args.epochs - 1:
            print("  Recomputing features …", flush=True)
            Phi_tr, feat_scale = _encode_dataset(model, data_loaders["train"])
            Phi_val, _         = _encode_dataset(model, data_loaders["val"],
                                                 feat_scale=feat_scale)

    # Restore best classifier
    model.classifier.weight.values = best_W
    if model.classifier.bias is not None and best_b is not None:
        model.classifier.bias.values = best_b

    print(f"\nBest val accuracy: {100 * best_acc:.2f} %")
    return best_acc, best_scale


def _encode_dataset(model: CKNet, loader: NumpyDataLoader,
                    feat_scale: float = None):
    """
    Encode all samples in `loader` through the CKN and return
    (Phi : (N, D) float64, feat_scale : float).
    """
    feats = []
    for data, _ in loader:
        feats.append(model.representation(data).astype(np.float64))
    Phi = np.concatenate(feats)
    if feat_scale is None:
        feat_scale = Phi.std() + 1e-10
    Phi /= feat_scale
    return Phi, feat_scale


# ─────────────────────────────────────────────────────────────────────────────
#  Command-line arguments
# ─────────────────────────────────────────────────────────────────────────────

def load_args():
    parser = argparse.ArgumentParser(
        description="Supervised CKN – Kernel Methods Challenge")
    parser.add_argument("--datapath",          type=str,   default=DATA_DIR)
    parser.add_argument("--model",             type=str,   default="ckn9",
                        choices=list(SUPMODELS.keys()))
    parser.add_argument("--n_filters",         type=int,   default=128,
                        help="filters per layer (128=fast, 512=paper)")
    parser.add_argument("--batch_size",        type=int,   default=128)
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--lr",                type=float, default=10.0,
                        help="classifier learning rate")
    parser.add_argument("--lr_z",             type=float, default=0.01,
                        help="CKN filter learning rate")
    parser.add_argument("--alpha",             type=float, default=1.0,
                        help="regularisation (divided by N internally)")
    parser.add_argument("--loss",              type=str,   default="sq_hinge",
                        choices=list(LOSS.keys()))
    parser.add_argument("--sampling_patches",  type=int,   default=100_000,
                        help="patches for K-means init")
    parser.add_argument("--n_z_samples",       type=int,   default=1000,
                        help="images per epoch for Z backprop")
    parser.add_argument("--feat_recompute",    type=int,   default=5,
                        help="re-encode features every k epochs")
    parser.add_argument("--val_frac",          type=float, default=0.1,
                        help="fraction of Xtr held out for validation")
    parser.add_argument("--zca",           type=bool, default=True,
                        help="ZCA whitening Bool")
    parser.add_argument("--zca_reg",           type=float, default=0.1,
                        help="ZCA whitening regularisation")
    parser.add_argument("--output",            type=str,   default=OUTPUT_CSV)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Supervised CKN  –  Competition data  (Mairal, NIPS 2016)")
    print("=" * 65, flush=True)

    args = load_args()
    print(args, flush=True)

    # ── 1. Load data ─────────────────────────────────────────────
    print("\n[1] Loading data …", flush=True)
    Xtr_all, Ytr_all, Xte = create_dataset(args.datapath)
    N_all = len(Xtr_all)

    # Random validation split
    rng   = np.random.default_rng(42)
    N_val = max(1, int(N_all * args.val_frac))
    val_i = rng.choice(N_all, N_val, replace=False)
    tr_mask = np.ones(N_all, dtype=bool); tr_mask[val_i] = False
    Xtr, Ytr   = Xtr_all[tr_mask],  Ytr_all[tr_mask]
    Xval, Yval = Xtr_all[~tr_mask], Ytr_all[~tr_mask]
    print(f"  train={len(Xtr)}, val={len(Xval)}, test={len(Xte)}", flush=True)

    # ── 2. ZCA whitening ─────────────────────────────────────────
    if args.zca:
        print(f"\n[2] ZCA whitening (reg={args.zca_reg}) …", flush=True)
        zca  = ZCA().fit(Xtr, reg=args.zca_reg)
        Xtr  = zca.transform(Xtr)
        Xval = zca.transform(Xval)
        Xte  = zca.transform(Xte)

    # ── 3. Build model ───────────────────────────────────────────
    print(f"\n[3] Building model '{args.model}' ({args.n_filters} filters) …",
          flush=True)
    model = SUPMODELS[args.model](n_filters=args.n_filters, alpha=args.alpha)
    print(model, flush=True)
    print(f"  Parameters: {count_parameters(model):,}", flush=True)
    print(f"  Feature dimension D = {model.out_features}", flush=True)

    # ── 4. Data loaders ──────────────────────────────────────────
    train_loader = NumpyDataLoader(Xtr,  Ytr,  batch_size=args.batch_size, shuffle=True)
    val_loader   = NumpyDataLoader(Xval, Yval, batch_size=args.batch_size, shuffle=False)
    init_loader  = NumpyDataLoader(Xtr,  Ytr,  batch_size=args.batch_size, shuffle=True)
    data_loaders = {"train": train_loader, "val": val_loader, "init": init_loader}

    # ── 5. Train ─────────────────────────────────────────────────
    print("\n[4] Training …", flush=True)
    tic = time.time()
    best_acc, feat_scale = sup_train(model, data_loaders, args)
    print(f"\nTotal training time: {(time.time()-tic)/60:.1f} min", flush=True)

    # ── 6. Predict on test set ───────────────────────────────────
    print("\n[5] Predicting on test set …", flush=True)
    model.features.changemode("inf")

    test_loader  = NumpyDataLoader(Xte, batch_size=args.batch_size, shuffle=False)
    all_feats = []
    for batch in test_loader:
        all_feats.append(model.representation(batch).astype(np.float64))
    Phi_te = np.concatenate(all_feats) / feat_scale
    pred   = model.classifier(Phi_te).argmax(1).astype(int)

    # Distribution sanity check
    print("\n  Prediction distribution:")
    for c in range(model.nclass):
        print(f"    Class {c} : {countX(pred, c)}")

    # Validation accuracy with best weights
    Phi_val_enc = []
    for data, _ in val_loader:
        Phi_val_enc.append(model.representation(data).astype(np.float64))
    Phi_val_enc = np.concatenate(Phi_val_enc) / feat_scale
    val_pred = model.classifier(Phi_val_enc).argmax(1)
    print(f"\n  Final validation accuracy: "
          f"{100 * accuracy_score(Yval, val_pred):.2f} %", flush=True)

    # ── 7. Write submission ──────────────────────────────────────
    print(f"\n[6] Saving predictions to {args.output} …", flush=True)
    df = pd.DataFrame({"Prediction": pred})
    df.index += 1
    df.to_csv(args.output, index_label="Id")
    print(f"  Done. {len(pred)} predictions written.", flush=True)

    # Save model weights
    np.save("ckn_W.npy",          model.classifier.weight.values)
    np.save("ckn_feat_scale.npy", np.array([feat_scale]))
    np.savez("ckn_filters.npz",
             **{f"Z_layer{k}": l.Z
                for k, l in enumerate(model.features.ckn_layers)})
    print("  Weights saved: ckn_W.npy, ckn_filters.npz", flush=True)


if __name__ == "__main__":
    main()
