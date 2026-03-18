import numpy as np
import math
import scipy.optimize

from utils import (EPS, gaussian_filter, matrix_inverse_sqrt, spherical_kmeans_,
                   zca_whitening, normalize_np, im2col, conv2d_fast)
from kernels import KERNELS


class Parameters:
    def __init__(self, shape):
        self.values = np.random.randn(*shape)
        self.gradients = np.zeros(shape)
        self.requires_grad = True

    def zero_gradients(self):
        self.gradients.fill(0)

    @property
    def shape(self):
        return self.values.shape

    def __repr__(self):
        return f"Parameters(shape={self.values.shape})"


class CKNLayer:
    def __init__(self, in_channels, out_channels, filter_size,
                 subsampling, padding="SAME", dilation=1, groups=1,
                 subsampling_factor=1 / math.sqrt(2),
                 kernel_func="exp", kernel_args=0.5):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = (filter_size, filter_size)
        self.subsampling = subsampling
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_func = KERNELS[kernel_func]
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == "exp":
            kernel_args = [1. / ka ** 2 for ka in kernel_args]
        self.kernel_args = kernel_args
        self.kappa = lambda x: KERNELS[kernel_func](x, *self.kernel_args)
        self.mode = 'train'
        self.patch_dim = in_channels * filter_size * filter_size

        weight_shape = (out_channels, in_channels, filter_size, filter_size)
        self.weight = Parameters(weight_shape)
        self.bias = None
        self.parameters_liste = [self.weight]

        gauss = gaussian_filter(2 * subsampling + 1)
        self.pool_filter = np.outer(gauss, gauss).reshape(1, 1, 2 * subsampling + 1, 2 * subsampling + 1)
        self.pool_filter = np.broadcast_to(
            self.pool_filter, (out_channels, 1, 2 * subsampling + 1, 2 * subsampling + 1)).copy()

    def normalize(self):
        W = self.weight.values.reshape(self.out_channels, -1)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / np.maximum(norms, EPS)
        self.weight.values[:] = W.reshape(self.weight.values.shape)

    def compute_lintrans(self):
        W = self.weight.values.reshape(self.out_channels, -1)
        W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), EPS)
        M = np.dot(W, W.T)
        return matrix_inverse_sqrt(self.kappa(M))

    def _mult_layer(self, x_in, lintrans):
        batch_size, out_c, H, W = x_in.shape
        x_flat = x_in.reshape(batch_size, out_c, -1)
        x_out = np.einsum('ij,bjk->bik', lintrans, x_flat)
        return x_out.reshape(batch_size, out_c, H, W)

    def conv_layer(self, x):
        kH, kW = self.filter_size
        padding = kH // 2 if self.padding == "SAME" else 0
        N, C, H, W = x.shape

        cols = im2col(x, kH, kW, stride=1, padding=padding)
        patch_norms = np.linalg.norm(cols, axis=1, keepdims=True)
        cols_norm = cols / np.maximum(patch_norms, EPS)

        W_flat = self.weight.values.reshape(self.out_channels, -1)
        W_flat = W_flat / np.maximum(np.linalg.norm(W_flat, axis=1, keepdims=True), EPS)

        H_out = (H + 2 * padding - kH) // 1 + 1
        W_out = (W + 2 * padding - kW) // 1 + 1
        dot = np.tensordot(W_flat, cols_norm, axes=([1], [1])).transpose(1, 0, 2)
        dot = dot.reshape(N, self.out_channels, H_out, W_out)

        patch_norms_2d = patch_norms.reshape(N, 1, H_out, W_out)
        x_out = patch_norms_2d * self.kappa(dot)
        return x_out

    def pool_layer(self, x):
        return conv2d_fast(x, self.pool_filter,
                           stride=self.subsampling, padding=self.subsampling,
                           groups=self.out_channels)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.pool_layer(x)
        lintrans = self.compute_lintrans()
        return self._mult_layer(x, lintrans)

    def __call__(self, x):
        return self.forward(x)

    def extract_2d_patches(self, x):
        kH, kW = self.filter_size
        N, C, H, W = x.shape
        padding = kH // 2 if self.padding == "SAME" else 0
        cols = im2col(x, kH, kW, stride=1, padding=padding)
        return cols.transpose(0, 2, 1).reshape(-1, self.patch_dim)

    def sample_patches(self, x, n_sampling_patches=1000):
        patches = self.extract_2d_patches(x)
        idx = np.random.choice(patches.shape[0],
                               min(patches.shape[0], n_sampling_patches),
                               replace=False)
        return patches[idx]

    def unsup_train(self, patches, use_zca=True):
        patches = patches.copy()
        patches -= patches.mean(axis=0, keepdims=True)
        if use_zca and self.patch_dim > 2:
            patches, _ = zca_whitening(patches, eps=0.1)
        patches = normalize_np(patches)
        block_size = None if self.patch_dim < 1000 else 10 * self.patch_dim
        weight = spherical_kmeans_(patches, self.out_channels, block_size=block_size)[0]
        self.weight.values[:] = weight.reshape(self.weight.values.shape)

    def parameters(self):
        return self.parameters_liste

    def __repr__(self):
        return (f"CKNLayer(in={self.in_channels}, out={self.out_channels}, "
                f"filter={self.filter_size}, subsampling={self.subsampling})")


class LinearSVM:
    """
    Multiclass one-vs-rest linear SVM, squared hinge loss + L2 regularization.
    Optimized with L-BFGS-B (scipy).
    C = 1 / (alpha * n)
    """
    def __init__(self, in_features, out_features, alpha=1.0,
                 fit_bias=True, maxiter=1000):
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.fit_bias = fit_bias
        self.maxiter = maxiter

        self.weight = Parameters((out_features, in_features))
        if fit_bias:
            self.bias = Parameters((out_features,))
            self.bias.values = np.zeros(out_features)
        else:
            self.bias = None
        self.parameters_liste = [self.weight] + ([self.bias] if fit_bias else [])

    def forward(self, x):
        out = x @ self.weight.values.T
        if self.bias is not None:
            out = out + self.bias.values
        return out

    @staticmethod
    def _sq_hinge_loss_grad(w, b, x, y_binary, C):
        n = x.shape[0]
        scores = x @ w + b
        margins = 1.0 - y_binary * scores
        mask = margins > 0

        loss = 0.5 * np.dot(w, w) + C * np.sum(margins[mask] ** 2)
        active = mask * (-2.0 * C * y_binary * margins)
        grad_w = w + x.T @ active
        grad_b = active.sum()
        return loss, grad_w, grad_b

    def fit(self, x, y):
        n, D = x.shape
        n_class = self.out_features
        C = 1.0 / max(self.alpha * n, 1e-12)

        if self.fit_bias:
            scale_bias = np.sqrt(np.mean(x ** 2)) if x.size > 0 else 1.0
            scale_bias = max(scale_bias, EPS)
            x_aug = np.hstack([x, np.full((n, 1), scale_bias)])
            D_aug = D + 1
        else:
            x_aug = x
            D_aug = D
            scale_bias = 1.0

        W_all = np.zeros((n_class, D_aug), dtype=np.float64)

        for c in range(n_class):
            y_c = np.where(y == c, 1.0, -1.0)

            def obj(w_flat):
                w_flat = w_flat.astype(np.float64)
                if self.fit_bias:
                    w_c, b_c = w_flat[:-1], w_flat[-1] * scale_bias
                else:
                    w_c, b_c = w_flat, 0.0
                loss, gw, gb = self._sq_hinge_loss_grad(w_c, b_c, x, y_c, C)
                if self.fit_bias:
                    grad = np.append(gw, gb * scale_bias)
                else:
                    grad = gw
                return loss, grad.astype(np.float64)

            w0 = np.zeros(D_aug, dtype=np.float64)
            result = scipy.optimize.minimize(
                obj, w0, jac=True, method='L-BFGS-B',
                options={'maxiter': self.maxiter}
            )
            W_all[c] = result.x

        if self.fit_bias:
            self.weight.values = W_all[:, :-1].copy()
            self.bias.values = (W_all[:, -1] * scale_bias).copy()
        else:
            self.weight.values = W_all.copy()

        self.real_alpha = self.alpha

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.parameters_liste
