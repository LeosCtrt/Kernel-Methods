"""
crammer_singer.py — Multiclass Kernel SVM (Crammer & Singer, JMLR 2001).

Public API
----------
CrammerSingerSVM              — core dual solver (standalone)
CrammerSingerSVMClassifier    — drop-in replacement for LinearSVM inside CKNet
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ── kernel functions ──────────────────────────────────────────────────────────

def _linear_kernel(X1, X2):
    return X1 @ X2.T

def _rbf_kernel(X1, X2, gamma=1.0):
    return np.exp(-gamma * cdist(X1, X2, "sqeuclidean"))

def _poly_kernel(X1, X2, degree=3, coef0=1.0):
    return (X1 @ X2.T + coef0) ** degree


# ── thin Parameters shim ──────────────────────────────────────────────────────

class _ReadOnlyParameters:
    """
    Minimal stand-in for Parameters so that CKNet.get_parameters() works.
    The Crammer-Singer dual variables are not gradient-based; this stub
    presents a zero-filled array and sets requires_grad=False.
    """
    def __init__(self, shape):
        self.values = np.zeros(shape)
        self.gradients = np.zeros(shape)
        self.requires_grad = False

    def zero_gradients(self):
        self.gradients.fill(0)

    @property
    def shape(self):
        return self.values.shape

    def __repr__(self):
        return f"_ReadOnlyParameters(shape={self.values.shape}, requires_grad=False)"


# ── fixed-point subroutine (Section 6, Figure 3 of Crammer & Singer) ─────────

def _fixed_point(D: np.ndarray, theta_init: float,
                 eps: float = 1e-6, max_iter: int = 10_000) -> np.ndarray:
    """Return ν_r = min{θ*, D_r} where θ* solves Σ min{θ, D_r} = ΣD_r - 1."""
    k = len(D)
    theta = float(theta_init)
    for _ in range(max_iter):
        theta_new = float(np.mean(np.maximum(theta, D)) - 1.0 / k)
        if abs(theta_new - theta) / (abs(theta) + 1e-12) <= eps:
            theta = theta_new
            break
        theta = theta_new
    return np.minimum(theta, D)


# ── core Crammer-Singer SVM ───────────────────────────────────────────────────

class CrammerSingerSVM:
    """Multiclass Kernel SVM — Crammer & Singer (JMLR 2001) dual solver."""

    def __init__(self, beta=1.0, kernel="rbf", gamma=1.0,
                 degree=3, coef0=1.0, epsilon=0.001,
                 max_iter=10_000, use_cooling=True,
                 use_active_set=True, verbose=False):
        self.beta           = beta
        self.kernel         = kernel
        self.gamma          = gamma
        self.degree         = degree
        self.coef0          = coef0
        self.epsilon        = epsilon
        self.max_iter       = max_iter
        self.use_cooling    = use_cooling
        self.use_active_set = use_active_set
        self.verbose        = verbose

    def _K(self, X1, X2):
        if self.kernel == "linear":
            return _linear_kernel(X1, X2)
        elif self.kernel == "rbf":
            return _rbf_kernel(X1, X2, gamma=self.gamma)
        elif self.kernel == "poly":
            return _poly_kernel(X1, X2, degree=self.degree, coef0=self.coef0)
        raise ValueError(f"Unknown kernel '{self.kernel}'.")

    def _psi_all(self, F, tau, y_idx, k):
        """KKT violation ψ_i (Eq. 32)."""
        m = F.shape[0]
        delta = np.zeros((m, k))
        delta[np.arange(m), y_idx] = 1.0
        mask  = tau < delta
        max_F = F.max(axis=1)
        F_inf = np.where(mask, F, np.inf)
        min_F = F_inf.min(axis=1)
        return max_F - min_F

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        m = X.shape[0]

        self.classes_ = np.unique(y)
        k = len(self.classes_)
        lbl2idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx   = np.array([lbl2idx[yi] for yi in y], dtype=int)

        K  = self._K(X, X)
        A  = np.diag(K).copy()

        tau = np.zeros((m, k))
        F   = np.zeros((m, k))
        F[np.arange(m), y_idx] = -self.beta

        delta_mat = np.zeros((m, k))
        delta_mat[np.arange(m), y_idx] = 1.0

        active = np.zeros(m, dtype=bool) if self.use_active_set else np.ones(m, dtype=bool)
        eps_0  = 0.999

        for t in range(self.max_iter):
            eps_t = max(self.epsilon, eps_0 / np.log10(t + 10.0)) \
                    if self.use_cooling else self.epsilon

            p = None
            if self.use_active_set:
                ai = np.where(active)[0]
                if len(ai):
                    psi_a = self._psi_all(F[ai], tau[ai], y_idx[ai], k)
                    best  = int(psi_a.argmax())
                    if psi_a[best] > eps_t:
                        p = int(ai[best])
                if p is None:
                    ii = np.where(~active)[0]
                    if not len(ii):
                        break
                    psi_i = self._psi_all(F[ii], tau[ii], y_idx[ii], k)
                    best  = int(psi_i.argmax())
                    if psi_i[best] <= self.epsilon:
                        break
                    p = int(ii[best])
                    active[p] = True
            else:
                psi = self._psi_all(F, tau, y_idx, k)
                p   = int(psi.argmax())
                if psi[p] < self.epsilon:
                    break

            D  = F[p] / A[p] - tau[p] + delta_mat[p]
            nu = _fixed_point(D, float(np.mean(D) - 1.0 / k), eps=eps_t / 2.0)

            tau_new   = nu - D + delta_mat[p]
            delta_tau = tau_new - tau[p]
            F        += np.outer(K[:, p], delta_tau)
            tau[p]    = tau_new

            if self.verbose and t % 500 == 0:
                psi_max = self._psi_all(F, tau, y_idx, k).max()
                print(f"  iter {t:6d}  max ψ={psi_max:.5f}  ε_t={eps_t:.4f}  "
                      f"|A|={active.sum()}")

        self.tau_       = tau
        self.X_train_   = X
        self.n_iter_    = t + 1
        self.n_support_ = int((np.abs(tau).sum(axis=1) > 1e-10).sum())

        if self.verbose:
            print(f"Done — {self.n_iter_} iters, {self.n_support_}/{m} support patterns.")
        return self

    def decision_function(self, X):
        """Returns (n_samples, n_classes) score matrix."""
        return self._K(np.asarray(X, dtype=float), self.X_train_) @ self.tau_

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


# ── drop-in wrapper for CKNet ─────────────────────────────────────────────────

class CrammerSingerSVMClassifier:
    """
    Drop-in replacement for LinearSVM inside CKNet.

    Constructor mirrors LinearSVM's signature plus Crammer-Singer knobs:

        CrammerSingerSVMClassifier(
            in_features, out_features,
            alpha=1.0, fit_bias=True, maxiter=10_000,
            kernel='rbf', gamma=1.0, degree=3, coef0=1.0,
            epsilon=0.001, use_cooling=True, use_active_set=True,
            verbose=False,
        )

    Notes
    -----
    * `alpha` is converted to `beta = 1/alpha` (larger α → more
      regularisation → smaller β), consistent with LinearSVM.
    * `fit_bias` is accepted but ignored: kernel methods handle the
      intercept implicitly. Pre-centre features if using a linear kernel.
    * `parameters()` returns stub _ReadOnlyParameters (requires_grad=False)
      so CKNet.get_parameters() does not crash.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
        fit_bias: bool = True,
        maxiter: int = 10_000,
        kernel: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        epsilon: float = 0.001,
        use_cooling: bool = True,
        use_active_set: bool = True,
        verbose: bool = False,
    ):
        self.in_features  = in_features
        self.out_features = out_features
        self.alpha        = alpha
        self.fit_bias     = fit_bias

        beta = 1.0 / max(alpha, 1e-12)

        self._svm = CrammerSingerSVM(
            beta           = beta,
            kernel         = kernel,
            gamma          = gamma,
            degree         = degree,
            coef0          = coef0,
            epsilon        = epsilon,
            max_iter       = maxiter,
            use_cooling    = use_cooling,
            use_active_set = use_active_set,
            verbose        = verbose,
        )

        # Stubs so CKNet.get_parameters() works unchanged
        self.weight = _ReadOnlyParameters((out_features, in_features))
        self.bias   = _ReadOnlyParameters((out_features,))
        self.parameters_liste = [self.weight, self.bias]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._svm.fit(x, y)
        self.n_support_ = self._svm.n_support_

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Raw class scores (N, n_classes). Use argmax for predictions."""
        return self._svm.decision_function(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def parameters(self):
        return self.parameters_liste

    def __repr__(self):
        svm = self._svm
        return (
            f"CrammerSingerSVMClassifier("
            f"in={self.in_features}, out={self.out_features}, "
            f"alpha={self.alpha}, kernel={svm.kernel!r}, "
            f"gamma={svm.gamma}, beta={svm.beta:.4f})"
        )
