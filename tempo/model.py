import numpy as np
from layering import CKNLayer, Linear

# ─────────────────────────────────────────────────────────────────────────────
#  CKNSequential  —  ordered stack of CKNLayer objects
# ─────────────────────────────────────────────────────────────────────────────

class CKNSequential:
    """
    Sequential container for CKN layers, mirroring CKNSequential from
    the other codebase.

    Args:
        in_channels       (int):  Input channels (e.g. 3 for RGB).
        out_channels_list (list): Output channels per layer.
        filter_sizes      (list): Patch size per layer.
        subsamplings      (list): Pooling subsampling factor per layer.
        kernel_funcs      (list, optional): Kernel name per layer.
        kernel_args_list  (list, optional): Kernel α per layer.
    """

    def __init__(self, in_channels: int,
                 out_channels_list: list,
                 filter_sizes:      list,
                 subsamplings:      list,
                 kernel_funcs:      list = None,
                 kernel_args_list:  list = None):

        assert len(out_channels_list) == len(filter_sizes) == len(subsamplings), \
            "out_channels_list, filter_sizes and subsamplings must have the same length"

        self.n_layers   = len(out_channels_list)
        self.in_channels  = in_channels
        self.out_channels = out_channels_list[-1]

        self.ckn_layers: list[CKNLayer] = []
        C = in_channels

        for i in range(self.n_layers):
            kf   = "exp"  if kernel_funcs     is None else kernel_funcs[i]
            ka   = 4.0    if kernel_args_list  is None else kernel_args_list[i]
            layer = CKNLayer(C, out_channels_list[i],
                             filter_sizes[i], subsamplings[i],
                             kernel_func=kf, kernel_args=ka)
            self.ckn_layers.append(layer)
            C = out_channels_list[i]

    # ── modes ────────────────────────────────────────────────────

    def changemode(self, mode: str = "inf"):
        """Switch all layers between 'train' and 'inf' modes."""
        for layer in self.ckn_layers:
            layer.mode = mode

    # ── forward pass ─────────────────────────────────────────────

    def forward(self, x: np.ndarray):
        """
        Forward pass for a batch x : (B, C, H, W).
        Returns output : (B, p, H_out, W_out).
        """
        out = []
        for img in x:
            I = img
            for layer in self.ckn_layers:
                I, _ = layer.forward(I)
            out.append(I)
        return np.stack(out)

    def forward_single(self, img: np.ndarray):
        """Forward pass for a single image img : (C, H, W).  Returns (I_out, caches)."""
        I, caches = img, []
        for layer in self.ckn_layers:
            I, c = layer.forward(I)
            caches.append(c)
        return I, caches

    def forward_at(self, x: np.ndarray, i: int = 0):
        """Forward through layer i only, for a batch x."""
        out = []
        for img in x:
            I, _ = self.ckn_layers[i].forward(img)
            out.append(I)
        return np.stack(out)

    def representation(self, x: np.ndarray, n: int = None):
        """
        Forward through the first `n` layers (default: all layers).
        Returns batch output (B, C_n, H_n, W_n).
        """
        if n is None or n < 0:
            n = self.n_layers
        out = []
        for img in x:
            I = img
            for i in range(n):
                I, _ = self.ckn_layers[i].forward(I)
            out.append(I)
        return np.stack(out)

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    # ── normalisation ─────────────────────────────────────────────

    def normalize(self):
        for layer in self.ckn_layers:
            layer.normalize()

    # ── unsupervised initialisation ───────────────────────────────

    def unsup_train_(self, data_loader, n_sampling_patches: int = 100_000):
        """
        Layer-wise spherical K-means initialisation.

        For each layer j:
          1. Propagate data through layers 0 … j-1.
          2. Collect patches at layer j.
          3. Run spherical K-means and set filters.

        Args:
            data_loader:          Iterable yielding (X_batch, y_batch).
            n_sampling_patches:   Total patches to collect per layer.
        """
        for i, layer in enumerate(self.ckn_layers):
            print(f"\n  Training layer {i + 1}/{self.n_layers} "
                  f"(k={layer.out_channels}) …", flush=True)

            try:
                n_batches = len(data_loader)
            except TypeError:
                n_batches = 1
            n_per_batch = max(1, (n_sampling_patches + n_batches - 1) // n_batches)

            patches   = []
            n_patches = 0

            for data, _ in data_loader:
                # propagate through previous layers
                inter = self.representation(data, n=i)
                # collect patches from this layer
                batch_patches = layer.sample_patches(inter, n_per_batch)
                take = min(batch_patches.shape[0],
                           n_sampling_patches - n_patches)
                patches.append(batch_patches[:take])
                n_patches += take
                if n_patches >= n_sampling_patches:
                    break

            all_patches = np.concatenate(patches, axis=0)
            layer.unsup_train(all_patches)

        print("  Unsupervised CKN init done.", flush=True)

    # ── parameter access ──────────────────────────────────────────

    def parameters(self) -> list:
        params = []
        for layer in self.ckn_layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        lines = ["CKNSequential("]
        for i, l in enumerate(self.ckn_layers):
            lines.append(f"  ({i}): {l}")
        lines.append(")")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  CKNet  —  full model: CKNSequential + Linear classifier head
# ─────────────────────────────────────────────────────────────────────────────

class CKNet:
    """
    Convolutional Kernel Network with a linear classification head.

    The representation (CKN feature extractor) and classifier can be trained
    jointly (end-to-end, via backprop through every layer) or the classifier
    alone (unsup_train_classifier).

    Args:
        nclass            (int):   Number of output classes.
        in_channels       (int):   Input channels (3 for RGB).
        out_channels_list (list):  Filters per layer.
        kernel_sizes      (list):  Patch size per layer.
        subsamplings      (list):  Subsampling factor per layer.
        kernel_funcs      (list):  Kernel name per layer.
        kernel_args_list  (list):  Kernel α per layer.
        image_size        (int):   Spatial size of input images (default 32).
        fit_bias          (bool):  Include bias in classifier.
        alpha             (float): Regularisation strength for classifier.
    """

    def __init__(self, nclass: int, in_channels: int,
                 out_channels_list: list, kernel_sizes: list,
                 subsamplings: list, kernel_funcs: list = None,
                 kernel_args_list: list = None,
                 image_size: int = 32, fit_bias: bool = True,
                 alpha: float = 0.0):

        self.nclass  = nclass
        self.alpha   = alpha

        self.features = CKNSequential(
            in_channels, out_channels_list, kernel_sizes, subsamplings,
            kernel_funcs, kernel_args_list)

        # Infer output feature dimension by running a dummy forward pass
        dummy           = np.zeros((1, in_channels, image_size, image_size),
                                   dtype=np.float32)
        dummy_out       = self.features.forward(dummy)
        self.out_features = int(np.prod(dummy_out.shape[1:]))

        self.classifier = Linear(self.out_features, nclass, fit_bias=fit_bias)

    # ── inference ─────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass: features + linear head.  x : (B, C, H, W)."""
        return self.classifier(self.representation(x))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def representation(self, x: np.ndarray) -> np.ndarray:
        """CKN features only, flattened.  Returns (B, D)."""
        return self.features.forward(x).reshape(len(x), -1)

    # ── unsupervised (K-means) initialisation ────────────────────

    def unsup_train_ckn(self, data_loader, n_sampling_patches: int = 100_000):
        """Layer-wise spherical K-means init of all CKN filters."""
        self.features.unsup_train_(data_loader, n_sampling_patches)

    # ── supervised training of the classifier head only ──────────

    def unsup_train_classifier(self, data_loader):
        """
        Encode the full training set then fit the linear classifier using the
        accumulated features.  ('unsup' here means no backprop through the CKN.)
        """
        feats, targets = self._collect_representations(data_loader)
        # Simple closed-form ridge regression via SGD warm-start
        self.classifier.weight.values = np.zeros_like(self.classifier.weight.values)
        if self.classifier.bias is not None:
            self.classifier.bias.values = np.zeros_like(self.classifier.bias.values)
        self._fit_linear_sgd(feats, targets)

    def _fit_linear_sgd(self, feats: np.ndarray, targets: np.ndarray,
                        n_iter: int = 200, lr: float = 0.1):
        """Mini internal SGD for classifier-only warm-start."""
        from loss import SquaredHingeLoss
        N, D = feats.shape
        lam  = self.alpha / N
        for _ in range(n_iter):
            perm = np.random.permutation(N)
            for s in range(0, N, 256):
                idx = perm[s:s + 256]
                Pb  = feats[idx]
                yb  = targets[idx]
                sc  = self.classifier(Pb)
                C_y = self.nclass
                bs  = len(idx)
                Y   = -np.ones((bs, C_y))
                Y[np.arange(bs), yb.astype(int)] = 1.0
                m   = np.maximum(0.0, 1.0 - Y * sc)
                dW  = (-2.0 * Y * m / (bs * C_y)) .T @ Pb + lam * self.classifier.weight.values
                self.classifier.weight.values -= lr * dW
                if self.classifier.bias is not None:
                    db  = (-2.0 * Y * m / (bs * C_y)).sum(0)
                    self.classifier.bias.values -= lr * db.reshape(-1, 1)

    def _collect_representations(self, data_loader):
        """Encode entire data_loader into (feats, targets) arrays."""
        all_feats, all_targets = [], []
        for data, target in data_loader:
            all_feats.append(self.representation(data))
            all_targets.append(target)
        return np.concatenate(all_feats), np.concatenate(all_targets)

    # ── normalisation ─────────────────────────────────────────────

    def normalize(self):
        self.features.normalize()

    # ── parameter access ──────────────────────────────────────────

    def get_parameters(self) -> list:
        params = list(self.features.parameters())
        params.extend(self.classifier.parameters())
        return params

    def __repr__(self):
        return (f"CKNet(\n"
                f"  features = {self.features}\n"
                f"  classifier = {self.classifier}\n"
                f"  out_features = {self.out_features}\n"
                f")")


# ─────────────────────────────────────────────────────────────────────────────
#  Named architectures  (matching the paper and the other codebase)
# ─────────────────────────────────────────────────────────────────────────────

class SupCKNetCifar_9L(CKNet):
    """
    9-layer CKN matching the paper architecture (Mairal NIPS 2016, §4.1):
      • Layers 1,3,5,7,9 : 3×3 patches,  subsampling √2  (layer 9 → ×3)
      • Layers 2,4,6,8   : 1×1 patches,  no subsampling
      • 512 filters / layer,  α = 4  (or 128 for faster run)
    """
    def __init__(self, n_filters: int = 512, alpha: float = 0.0, **kwargs):
        s = float(np.sqrt(2))
        filter_sizes     = [3, 1, 3, 1, 3, 1, 3, 1, 3]
        out_channels     = [n_filters] * 9
        subsamplings     = [s, 1, s, 1, s, 1, s, 1, 3.0]
        kernel_funcs     = ["exp"] * 9
        kernel_args_list = [4.0] * 9      # α = 1 / 0.5² = 4
        super().__init__(
            10, 3, out_channels, filter_sizes, subsamplings,
            kernel_funcs=kernel_funcs, kernel_args_list=kernel_args_list,
            alpha=alpha, **kwargs)


class SupCKNetCifar_3L(CKNet):
    """
    Lightweight 3-layer CKN (fast baseline, ~35–40 % on 5k samples).
    """
    def __init__(self, n_filters: int = 128, alpha: float = 0.0, **kwargs):
        super().__init__(
            10, 3,
            out_channels_list = [n_filters, n_filters, n_filters],
            kernel_sizes      = [3, 3, 3],
            subsamplings      = [2, 2, 2],
            kernel_funcs      = ["exp", "exp", "exp"],
            kernel_args_list  = [4.0, 4.0, 4.0],
            alpha=alpha, **kwargs)


class SupCKNetCifar_5L(CKNet):
    """
    5-layer CKN with alternating 3×3 / 1×1 layers (matches ckn5 from the
    other codebase but with the paper's RBF kernel and backprop).
    """
    def __init__(self, n_filters: int = 128, alpha: float = 0.0, **kwargs):
        super().__init__(
            10, 3,
            out_channels_list = [n_filters, n_filters, n_filters,
                                 n_filters, n_filters],
            kernel_sizes      = [3, 1, 3, 1, 3],
            subsamplings      = [float(np.sqrt(2)), 1,
                                 float(np.sqrt(2)), 1, 3.0],
            kernel_funcs      = ["exp"] * 5,
            kernel_args_list  = [4.0] * 5,
            alpha=alpha, **kwargs)


SUPMODELS = {
    "ckn9": SupCKNetCifar_9L,   # paper architecture
    "ckn5": SupCKNetCifar_5L,   # medium
    "ckn3": SupCKNetCifar_3L,   # fast baseline
}
