import numpy as np
from kernels import kernels as KERNELS
from utils import (
    EPS, EPS_NORM, EPS_REG,
    compute_A_matrices,
    im2col, col2im,
    pool_forward, pool_backward,
    spherical_kmeans, normalize_np,
)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters  —  PyTorch-style trainable tensor wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Parameters:
    """
    Trainable parameter with associated gradient buffer.
    Mirrors PyTorch's nn.Parameter interface.
    """

    def __init__(self, shape):
        self.values        = np.random.randn(*shape)
        self.gradients     = np.zeros(shape)
        self.requires_grad = True

    def zero_gradients(self):
        self.gradients.fill(0)

    @property
    def shape(self):
        return self.values.shape

    def __repr__(self):
        return f"Parameters(shape={self.values.shape})"


# ─────────────────────────────────────────────────────────────────────────────
# CKNLayer  —  one layer of a Convolutional Kernel Network
#
#  Forward  (Eq. 3 + §2):
#    E     = im2col(I_{j-1})                      patch matrix (d, n)
#    S     = diag(‖E_l‖)                          patch norms
#    E_n   = E / S                                unit-norm patches
#    M     = A κ(Zᵀ E_n) S                       pre-pool feature map
#    I_j   = pool(M, s)                           pooled output
#
#  Backward  (Proposition 1, Mairal NIPS 2016):
#    B_j   = κ'(Zᵀ E_n) ⊙ (A U Pᵀ)
#    C_j   = A^{1/2} I_j Uᵀ A^{3/2}
#    g_j(U)= E B_j^T − ½ Z (κ'(ZᵀZ) ⊙ (C_j + C_j^T))   ← dZ
#    h_j(U)= E_j*[ Z B_j + E (S^{-2} ⊙ diag(Mᵀ UPᵀ − Eᵀ Z B_j)) ]  ← dI_{j-1}
# ─────────────────────────────────────────────────────────────────────────────

class CKNLayer:
    """
    Single Convolutional Kernel Network layer.

    Stores filters Z : (d, p) with (approximately) unit-norm columns,
    and caches A, A^{1/2}, A^{3/2} for fast repeated forward/backward calls.

    Args:
        in_channels  (int):   Number of input feature-map channels.
        out_channels (int):   Number of filters  p  (= output channels).
        filter_size  (int):   Spatial patch size  e  (e × e patches).
        subsampling  (float): Gaussian pooling subsampling factor s.
        kernel_func  (str):   Key in `kernels` dict — "exp" or "poly".
        kernel_args  (float): α parameter for the chosen kernel.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 filter_size: int, subsampling: float,
                 kernel_func: str = "exp", kernel_args: float = 4.0):

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.filter_size  = filter_size
        self.subsampling  = subsampling
        self.kernel_func  = kernel_func
        self.alpha        = float(kernel_args)

        self.kappa   = KERNELS[kernel_func]["fn"]
        self.kappa_d = KERNELS[kernel_func]["deriv"]

        d = in_channels * filter_size * filter_size
        self.Z   = np.zeros((d, out_channels), dtype=np.float64)
        self.vel = np.zeros_like(self.Z)   # momentum buffer for projected SGD
        self.A = self.A_h = self.A_3h = None

        self._init_random()

    # ── initialisation ────────────────────────────────────────────

    def _init_random(self):
        d = self.Z.shape[0]
        Z = np.random.randn(d, self.out_channels)
        Z /= np.linalg.norm(Z, axis=0, keepdims=True) + 1e-10
        self.set_Z(Z)

    def set_Z(self, Z: np.ndarray):
        """
        Store Z and recompute cached A matrices.
        Caller is responsible for ensuring unit-norm columns and absence of NaN/Inf.
        """
        self.Z = Z.astype(np.float64)
        self.A, self.A_h, self.A_3h = compute_A_matrices(
            self.Z, self.kappa, self.alpha, EPS_REG)

    @staticmethod
    def _project_unit_sphere(Z: np.ndarray,
                             vel: np.ndarray = None) -> np.ndarray:
        """
        Project each column of Z onto the unit sphere.
        Resets momentum for any column that was non-finite.
        """
        bad = ~np.isfinite(Z).all(axis=0)
        if bad.any():
            rnd = np.random.randn(Z.shape[0], int(bad.sum()))
            rnd /= np.linalg.norm(rnd, axis=0, keepdims=True) + 1e-10
            Z[:, bad] = rnd
            if vel is not None:
                vel[:, bad] = 0.0
        Z /= np.linalg.norm(Z, axis=0, keepdims=True) + 1e-10
        return Z

    def normalize(self):
        """Re-project each filter column onto the unit sphere."""
        col_norms = np.linalg.norm(self.Z, axis=0, keepdims=True) + 1e-10
        self.set_Z(self.Z / col_norms)

    # ── unsupervised initialisation via spherical K-means ────────

    def unsup_train(self, patches: np.ndarray):
        """
        Fit filters to `patches` (N, d) using spherical K-means.
        Mirrors the unsup_train method of the other codebase.
        """
        patches = normalize_np(patches.copy())
        centroids = spherical_kmeans(patches, self.out_channels, n_iter=10)
        self.set_Z(centroids.T)   # centroids : (k, d)  →  Z : (d, k)

    def sample_patches(self, x_in: np.ndarray,
                       n_sampling_patches: int = 1000) -> np.ndarray:
        """
        Extract and randomly subsample patches from a batch x_in : (B, C, H, W).
        Returns array of shape (n_take, d).
        """
        patches = []
        for img in x_in:
            E = im2col(img, self.filter_size)   # (d, H*W)
            patches.append(E.T)
        patches = np.concatenate(patches, axis=0)   # (B*H*W, d)
        n_take  = min(patches.shape[0], n_sampling_patches)
        idx     = np.random.choice(patches.shape[0], n_take, replace=False)
        return patches[idx]

    # ── forward pass ─────────────────────────────────────────────

    def forward(self, I_prev: np.ndarray):
        """
        Forward pass through the layer.

        Args:
            I_prev (ndarray): Input feature map, shape (C, H, W)  [single image].

        Returns:
            I_out  (ndarray): Output feature map, shape (p, H_out, W_out).
            cache  (dict):    Everything needed for backward().
        """
        C, H, W = I_prev.shape
        p, e    = self.out_channels, self.filter_size

        E       = im2col(I_prev, e)                                  # (d, n)
        norms   = np.sqrt((E ** 2).sum(0)) + EPS_NORM                # (n,)
        En      = E / norms                                          # (d, n)
        ZtEn    = self.Z.T @ En                                      # (p, n)
        kZtEn   = self.kappa(ZtEn, self.alpha)                       # (p, n)
        M       = (self.A @ kZtEn) * norms                           # (p, n)
        Msp     = M.reshape(p, H, W)

        I_out, h_idx, w_idx, Hin, Win = pool_forward(Msp, self.subsampling)

        cache = dict(I_prev=I_prev, E=E, norms=norms, En=En,
                     ZtEn=ZtEn, M=M, I_out=I_out,
                     h_idx=h_idx, w_idx=w_idx, Hin=Hin, Win=Win)
        return I_out, cache

    def __call__(self, I_prev: np.ndarray):
        return self.forward(I_prev)

    # ── backward pass  (Proposition 1) ───────────────────────────

    def backward(self, dI_out: np.ndarray, cache: dict):
        """
        Backward pass: compute gradients w.r.t. filters Z and input I_{j-1}.

        Args:
            dI_out (ndarray): Upstream gradient, shape (p, H_out, W_out).
            cache  (dict):    Cache returned by forward().

        Returns:
            dZ      (ndarray): Gradient w.r.t. Z,       shape (d, p).
            dI_prev (ndarray): Gradient w.r.t. I_{j-1}, shape (C, H, W).
        """
        E      = cache["E"];      norms  = cache["norms"]
        En     = cache["En"];     ZtEn   = cache["ZtEn"]
        M      = cache["M"];      I_out  = cache["I_out"]
        h_idx  = cache["h_idx"];  w_idx  = cache["w_idx"]
        Hin    = cache["Hin"];    Win    = cache["Win"]
        C, H_in, W_in = cache["I_prev"].shape
        p, n          = self.out_channels, H_in * W_in

        # Upsample dI_out → dM  (= U Pᵀ in the paper)
        if self.subsampling > 1.0:
            dM = pool_backward(dI_out, h_idx, w_idx, Hin, Win,
                               self.subsampling).reshape(p, n)
        else:
            dM = dI_out.reshape(p, n).copy()

        # ── B_j = κ'(Zᵀ E_n) ⊙ (A U Pᵀ) ────────────────────────
        B = self.kappa_d(ZtEn, self.alpha) * (self.A @ dM)           # (p, n)

        # ── C_j = A^{1/2} I_j Uᵀ A^{3/2} ───────────────────────
        Ij_flat = I_out.reshape(p, -1)                                # (p, |Ω|)
        U_flat  = dI_out.reshape(p, -1)                               # (p, |Ω|)
        Cj      = self.A_h @ Ij_flat @ U_flat.T @ self.A_3h           # (p, p)

        # ── g_j(U): gradient w.r.t. Z_j ─────────────────────────
        ZtZ    = self.Z.T @ self.Z
        kd_ZtZ = self.kappa_d(ZtZ, self.alpha)
        dZ     = E @ B.T - 0.5 * self.Z @ (kd_ZtZ * (Cj + Cj.T))    # (d, p)

        # ── h_j(U): gradient w.r.t. I_{j-1} ─────────────────────
        ZB            = self.Z @ B                                    # (d, n)
        diag_MdM      = (M  * dM).sum(0)                             # (n,)
        diag_EZB      = (E  * ZB).sum(0)                             # (n,)
        d_vec         = (diag_MdM - diag_EZB) / norms ** 2           # (n,)
        inner         = ZB + E * d_vec                                # (d, n)
        dI_prev       = col2im(inner, C, H_in, W_in, self.filter_size)

        return dZ, dI_prev

    # ── filter update  (projected SGD on unit sphere) ────────────

    def step_Z(self, dZ: np.ndarray, lr: float,
               momentum: float = 0.9, grad_clip: float = 1.0):
        """
        Update filters via projected gradient descent with momentum.

        Args:
            dZ        (ndarray): Gradient w.r.t. Z, shape (d, p).
            lr        (float):   Learning rate.
            momentum  (float):   Momentum coefficient.
            grad_clip (float):   Per-column gradient L2-norm clip threshold.
        """
        # Sanitize and clip
        dZ        = np.nan_to_num(dZ, nan=0.0, posinf=0.0, neginf=0.0)
        col_gnorm = np.linalg.norm(dZ, axis=0, keepdims=True) + 1e-10
        dZ        = dZ * np.minimum(1.0, grad_clip / col_gnorm)

        self.vel  = momentum * self.vel - lr * dZ
        Z_new     = self.Z + self.vel

        # Project onto unit sphere
        col_norms = np.linalg.norm(Z_new, axis=0, keepdims=True) + 1e-10
        Z_proj    = Z_new / col_norms

        # Roll back any non-finite columns
        if not np.isfinite(Z_proj).all():
            bad = ~np.isfinite(Z_proj).all(axis=0)
            Z_proj[:, bad] = self.Z[:, bad]
            self.vel[:, bad] = 0.0

        self.set_Z(Z_proj)

    # ── parameter access ─────────────────────────────────────────

    def parameters(self) -> list:
        """Return list of Parameters objects (just the filter matrix here)."""
        p = Parameters(self.Z.shape)
        p.values    = self.Z
        p.gradients = np.zeros_like(self.Z)
        return [p]

    def __repr__(self):
        return (f"CKNLayer(in={self.in_channels}, out={self.out_channels}, "
                f"filter={self.filter_size}x{self.filter_size}, "
                f"s={self.subsampling:.2f}, kernel={self.kernel_func}(α={self.alpha}))")


# ─────────────────────────────────────────────────────────────────────────────
# Linear  —  linear classification head  (output = x Wᵀ + b)
# ─────────────────────────────────────────────────────────────────────────────

class Linear:
    """
    Linear layer used as the CKN classifier head.

    Args:
        in_features  (int):   Input dimension D.
        out_features (int):   Number of classes C.
        fit_bias     (bool):  Whether to include a bias term.
    """

    def __init__(self, in_features: int, out_features: int,
                 fit_bias: bool = True):
        self.in_features  = in_features
        self.out_features = out_features
        self.fit_bias     = fit_bias

        self.weight = Parameters((out_features, in_features))
        self.weight.values = np.zeros((out_features, in_features))

        if fit_bias:
            self.bias = Parameters((out_features, 1))
            self.bias.values = np.zeros((out_features, 1))
        else:
            self.bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x (ndarray): shape (N, D)
        Returns:
            ndarray: shape (N, C)
        """
        out = x @ self.weight.values.T
        if self.bias is not None:
            out = out + self.bias.values.ravel()
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def parameters(self) -> list:
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        return (f"Linear(in={self.in_features}, out={self.out_features}, "
                f"bias={self.fit_bias})")
