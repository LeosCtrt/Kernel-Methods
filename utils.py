import numpy as np
import math

EPS = 1e-6


def normalize_np(x, p=2, axis=-1):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    x /= np.maximum(norm, EPS)
    return x


def gaussian_filter(size, sigma=None):
    if size == 1:
        return np.ones(1)
    if sigma is None:
        sigma = (size - 1.) / (2. * math.sqrt(2))
    m = (size - 1) / 2.
    filt = np.arange(-m, m + 1)
    filt = np.exp(-filt ** 2 / (2. * sigma ** 2))
    return filt / np.sum(filt)


def matrix_inverse_sqrt(input, eps=1e-2):
    e, v = np.linalg.eigh(input)
    e = np.maximum(e, 0)
    e_rsqrt = 1.0 / np.sqrt(e + eps)
    return np.dot(v, np.dot(np.diag(e_rsqrt), v.T))


def spherical_kmeans_(x, n_clusters, max_iters=100, block_size=None, verbose=True, init=None):
    n_samples, n_features = x.shape
    if init is None:
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        clusters = x[indices]
    else:
        clusters = init
    prev_sim = np.inf
    tmp = np.empty(n_samples)
    assign = np.empty(n_samples, dtype=np.int64)
    if block_size is None or block_size == 0:
        block_size = n_samples
    for n_iter in range(max_iters):
        for i in range(0, n_samples, block_size):
            end_i = min(i + block_size, n_samples)
            cos_sim = np.dot(x[i:end_i], clusters.T)
            tmp[i:end_i] = np.max(cos_sim, axis=1)
            assign[i:end_i] = np.argmax(cos_sim, axis=1)
        sim = np.mean(tmp)
        if (n_iter + 1) % 10 == 0 and verbose:
            print(f"  Spherical kmeans iter {n_iter+1}, objective {sim:.4f}")
        for j in range(n_clusters):
            index = assign == j
            if np.sum(index) == 0:
                idx = np.argmin(tmp)
                clusters[j] = x[idx]
                tmp[idx] = 1.
            else:
                c = np.mean(x[index], axis=0)
                clusters[j] = c / np.linalg.norm(c)
        if np.abs(prev_sim - sim) / (np.abs(sim) + 1e-20) < 1e-4:
            break
        prev_sim = sim
    return clusters, assign


def accuracy_score(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


def zca_whitening(patches, eps=0.1):
    """ZCA whitening on a batch of patches (N, D)."""
    patches = patches - patches.mean(axis=0, keepdims=True)
    cov = np.dot(patches.T, patches) / patches.shape[0]
    e, v = np.linalg.eigh(cov)
    e = np.maximum(e, 0)
    W_zca = v @ np.diag(1.0 / np.sqrt(e + eps)) @ v.T
    return patches @ W_zca.T, W_zca


def im2col(input, kernel_h, kernel_w, stride=1, padding=0):
    """Transform input into patch matrix for convolution."""
    N, C, H, W = input.shape
    if padding > 0:
        input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_out = (H + 2 * padding - kernel_h) // stride + 1
    W_out = (W + 2 * padding - kernel_w) // stride + 1
    shape = (N, C, kernel_h, kernel_w, H_out, W_out)
    strides = (*input.strides[:2], *input.strides[2:],
               input.strides[2] * stride, input.strides[3] * stride)
    cols = np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)
    return cols.reshape(N, C * kernel_h * kernel_w, H_out * W_out)


def conv2d_fast(input, weight, bias=None, stride=1, padding=0, groups=1):
    """Conv2d via im2col + matmul."""
    if not isinstance(weight, np.ndarray):
        weight = weight.values
    N, C, H, W = input.shape
    out_channels, in_channels_g, kH, kW = weight.shape

    if groups == 1:
        cols = im2col(input, kH, kW, stride=stride, padding=padding)
        W_flat = weight.reshape(out_channels, -1)
        out = np.tensordot(W_flat, cols, axes=([1], [1])).transpose(1, 0, 2)
        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1
        out = out.reshape(N, out_channels, H_out, W_out)
    else:
        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1
        out = np.zeros((N, out_channels, H_out, W_out))
        c_in_g = C // groups
        c_out_g = out_channels // groups
        for g in range(groups):
            x_g = input[:, g * c_in_g:(g + 1) * c_in_g]
            w_g = weight[g * c_out_g:(g + 1) * c_out_g]
            cols = im2col(x_g, kH, kW, stride=stride, padding=padding)
            W_flat = w_g.reshape(c_out_g, -1)
            res = np.tensordot(W_flat, cols, axes=([1], [1])).transpose(1, 0, 2)
            out[:, g * c_out_g:(g + 1) * c_out_g] = res.reshape(N, c_out_g, H_out, W_out)

    if bias is not None:
        out += bias
    return out
