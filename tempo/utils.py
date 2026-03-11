import numpy as np
from scipy import linalg as la

EPS      = 1e-6    # general numerical floor
EPS_NORM = 1e-5    # added to patch norms before division
EPS_REG  = 1e-3    # regularisation added to κ(ZᵀZ) before inversion

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ZCA whitening
# ─────────────────────────────────────────────────────────────────────────────

class ZCA:
    """
    Global ZCA whitening:  x̃ = W (x − μ),  W = V Λ^{−1/2} Vᵀ.

    Approximates the local whitening of [Paulin et al. 2015] used in the paper.
    """

    def fit(self, X: np.ndarray, reg: float = 0.1) -> "ZCA":
        N    = len(X)
        Xf   = X.reshape(N, -1).astype(np.float64)
        self.mu    = Xf.mean(0)
        Xc         = Xf - self.mu
        S          = (Xc.T @ Xc) / N
        np.fill_diagonal(S, S.diagonal() + reg)
        vals, vecs = la.eigh(S)
        vals       = np.maximum(vals, 1e-12)
        self.W_zca = (vecs * vals ** -0.5) @ vecs.T
        self.shape = X.shape[1:]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        N  = len(X)
        Xf = X.reshape(N, -1).astype(np.float64) - self.mu
        return (Xf @ self.W_zca.T).reshape((N,) + self.shape).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  A = (κ(ZᵀZ) + ε I)^{−1/2}  and its fractional powers
# ─────────────────────────────────────────────────────────────────────────────

def compute_A_matrices(Z: np.ndarray, kappa_fn, alpha: float,
                       eps_reg: float = EPS_REG):
    """
    Compute A = (κ(ZᵀZ) + ε I)^{−1/2} and its powers A^{1/2}, A^{3/2}.

    Z : (d, p)  — filter matrix. Caller is responsible for ensuring unit-norm
                  columns (set_Z does this before calling here).
    Returns A, A_half, A_3half  each of shape (p, p).

    NaN / Inf columns are replaced with random unit vectors before computation.
    """
    Z = Z.copy()
    # Replace non-finite columns with random unit vectors
    bad = ~np.isfinite(Z).all(axis=0)
    if bad.any():
        rnd = np.random.randn(Z.shape[0], int(bad.sum()))
        rnd /= np.linalg.norm(rnd, axis=0, keepdims=True) + 1e-10
        Z[:, bad] = rnd

    ZtZ  = Z.T @ Z
    K    = kappa_fn(ZtZ, alpha) + eps_reg * np.eye(Z.shape[1])
    K    = np.nan_to_num(K, nan=eps_reg, posinf=1.0, neginf=0.0)

    vals, vecs = la.eigh(K)
    vals = np.maximum(vals, 1e-12)
    A      = (vecs * vals ** -0.50) @ vecs.T
    A_half = (vecs * vals ** -0.25) @ vecs.T
    A_3h   = (vecs * vals ** -0.75) @ vecs.T
    return A, A_half, A_3h


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Patch extraction  im2col / col2im  (E_j  and  E_j*)
# ─────────────────────────────────────────────────────────────────────────────

def im2col(I: np.ndarray, p: int) -> np.ndarray:
    """
    Extract all overlapping p×p patches from I : (C, H, W).
    Returns E : (C·p·p, H·W)  with zero-padding so spatial size is preserved.
    """
    C, H, W = I.shape
    if p == 1:
        return I.reshape(C, H * W).copy()
    pad = p // 2
    Ip  = np.pad(I, ((0, 0), (pad, pad), (pad, pad)))
    sh  = (C, p, p, H, W)
    st  = (Ip.strides[0], Ip.strides[1], Ip.strides[2],
           Ip.strides[1], Ip.strides[2])
    return np.lib.stride_tricks.as_strided(Ip, sh, st).reshape(C * p * p, H * W).copy()


def col2im(P: np.ndarray, C: int, H: int, W: int, p: int) -> np.ndarray:
    """
    Adjoint of im2col  (E_j*):  (C·p·p, H·W) → (C, H, W).
    Patch contributions are summed back to their source locations.
    """
    if p == 1:
        return P.reshape(C, H, W).copy()
    pad = p // 2
    out = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=P.dtype)
    Pr  = P.reshape(C, p, p, H, W)
    for i in range(p):
        for j in range(p):
            out[:, i:i + H, j:j + W] += Pr[:, i, j]
    return out[:, pad:pad + H, pad:pad + W]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Gaussian pooling  (forward + adjoint backward)
# ─────────────────────────────────────────────────────────────────────────────

def _gauss1d(sigma: float):
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return (g / g.sum()).astype(np.float64), r


def _sep_blur(X: np.ndarray, g: np.ndarray, r: int,
              H: int, W: int, pad_mode: str = "reflect") -> np.ndarray:
    """Separable 2-D Gaussian blur on (C, H, W)."""
    Xp = np.pad(X, ((0, 0), (r, r), (0, 0)), mode=pad_mode)
    Xh = sum(g[k] * Xp[:, k:k + H, :] for k in range(2 * r + 1))
    Xp = np.pad(Xh, ((0, 0), (0, 0), (r, r)), mode=pad_mode)
    return sum(g[k] * Xp[:, :, k:k + W] for k in range(2 * r + 1))


def pool_forward(M: np.ndarray, s: float):
    """
    Forward Gaussian pooling.
    M : (C, H, W)  →  I_out : (C, H_out, W_out)
    Also returns (h_idx, w_idx, H_in, W_in) needed for pool_backward.
    """
    C, H, W = M.shape
    if s <= 1.0:
        return M.copy(), np.arange(H), np.arange(W), H, W

    g, r   = _gauss1d(s / 2.0)
    Mf     = _sep_blur(M, g, r, H, W, pad_mode="reflect")
    H_out  = max(1, int(np.floor(H / s)))
    W_out  = max(1, int(np.floor(W / s)))
    h_idx  = np.minimum(np.round(np.arange(H_out) * s).astype(int), H - 1)
    w_idx  = np.minimum(np.round(np.arange(W_out) * s).astype(int), W - 1)
    return Mf[:, h_idx[:, None], w_idx[None, :]], h_idx, w_idx, H, W


def pool_backward(dI_out: np.ndarray,
                  h_idx: np.ndarray, w_idx: np.ndarray,
                  H_in: int, W_in: int, s: float) -> np.ndarray:
    """
    Adjoint of pool_forward.
    dI_out : (C, H_out, W_out)  →  dM : (C, H_in, W_in)
    """
    if s <= 1.0:
        return dI_out.copy()
    g, r = _gauss1d(s / 2.0)
    C    = dI_out.shape[0]
    dM   = np.zeros((C, H_in, W_in), dtype=dI_out.dtype)
    np.add.at(dM, (slice(None), h_idx[:, None], w_idx[None, :]), dI_out)
    return _sep_blur(dM, g, r, H_in, W_in, pad_mode="constant")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Spherical K-means  (unsupervised filter initialisation)
# ─────────────────────────────────────────────────────────────────────────────

def spherical_kmeans(patches: np.ndarray, k: int,
                     n_iter: int = 10, verbose: bool = False) -> np.ndarray:
    """
    Spherical K-means clustering.
    patches : (N, d)  — will be L2-normalised internally.
    Returns centroids : (k, d) with unit-norm rows.
    """
    N, d = patches.shape
    nrm  = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-10
    X    = patches / nrm

    idx      = np.random.choice(N, k, replace=False)
    C        = X[idx].copy()
    C       /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-10
    BS       = 50_000
    prev_sim = -np.inf

    for it in range(n_iter):
        assign = np.empty(N, dtype=np.int32)
        tmp    = np.empty(N)
        for s in range(0, N, BS):
            e_            = min(s + BS, N)
            cos            = X[s:e_] @ C.T
            tmp[s:e_]      = cos.max(1)
            assign[s:e_]   = cos.argmax(1)

        sim = tmp.mean()
        if verbose and (it + 1) % 5 == 0:
            print(f"    K-means iter {it + 1}, objective {sim:.4f}")

        Cnew = np.zeros((k, d), dtype=np.float64)
        for j in range(k):
            mask = assign == j
            if mask.any():
                Cnew[j] = X[mask].sum(0)
            else:
                Cnew[j] = X[np.argmin(tmp)]   # re-seed empty cluster
        nrm = np.linalg.norm(Cnew, axis=1, keepdims=True) + 1e-10
        C   = (Cnew / nrm).astype(np.float32)

        if abs(sim - prev_sim) / (abs(sim) + 1e-20) < 1e-4:
            break
        prev_sim = sim

    return C


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalise x along `axis` in-place."""
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    x   /= np.maximum(norm, EPS)
    return x


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (y_true == y_pred).mean()


def count_parameters(model) -> int:
    return sum(np.prod(p.values.shape) for p in model.get_parameters())


def countX(lst, x) -> int:
    return sum(1 for e in lst if e == x)
