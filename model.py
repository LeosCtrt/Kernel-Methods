import numpy as np
import math

from layers import CKNLayer, LinearSVM
from hog import HOG


class CKNSequential:
    def __init__(self, in_channels, out_channels_list, filter_sizes,
                 subsamplings, kernel_funcs=None, kernel_args_list=None, **kwargs):
        assert len(out_channels_list) == len(filter_sizes) == len(subsamplings)
        self.n_layers = len(out_channels_list)
        self.ckn_layers = []
        ch = in_channels
        for i in range(self.n_layers):
            kf = kernel_funcs[i] if kernel_funcs else "exp"
            ka = kernel_args_list[i] if kernel_args_list else 0.5
            layer = CKNLayer(ch, out_channels_list[i], filter_sizes[i],
                             subsamplings[i], kernel_func=kf, kernel_args=ka, **kwargs)
            self.ckn_layers.append(layer)
            ch = out_channels_list[i]

    def changemode(self, mode='inf'):
        for layer in self.ckn_layers:
            layer.mode = mode
        print(f'Mode -> {mode}')

    def forward(self, x):
        for layer in self.ckn_layers:
            x = layer(x)
        return x

    def representation(self, x, n=0):
        for i in range(n):
            x = self.ckn_layers[i](x)
        return x

    def normalize(self):
        for layer in self.ckn_layers:
            layer.normalize()

    def unsup_train_(self, data_loader, n_sampling_patches=100000):
        for i, layer in enumerate(self.ckn_layers):
            print(f'\n--- Training CKN layer {i+1}/{self.n_layers} ---')
            try:
                n_per_batch = (n_sampling_patches + len(data_loader) - 1) // len(data_loader)
            except:
                n_per_batch = 1000
            patches = np.zeros((n_sampling_patches, layer.patch_dim))
            n_patches = 0
            for data, _ in data_loader:
                data = self.representation(data, i)
                batch_patches = layer.sample_patches(data, n_per_batch)
                size = min(batch_patches.shape[0], n_sampling_patches - n_patches)
                patches[n_patches:n_patches + size] = batch_patches[:size]
                n_patches += size
                if n_patches >= n_sampling_patches:
                    break
            layer.unsup_train(patches[:n_patches])

    def __call__(self, x):
        return self.forward(x)


class CKNet:
    """
    CKN model with a pluggable classifier.

    Parameters
    ----------
    classifier_cls : class, optional
        Class to use as the classifier. Must share the LinearSVM constructor
        signature: (in_features, out_features, alpha, fit_bias, maxiter, **classifier_kwargs).
        Defaults to LinearSVM.
    classifier_kwargs : dict, optional
        Extra keyword arguments forwarded only to the classifier constructor
        (e.g. kernel, gamma for CrammerSingerSVMClassifier).
    use_hog, hog_* : HOG feature-extractor options.
    """

    def __init__(self, nclass, in_channels, out_channels_list, kernel_sizes,
                 subsamplings, kernel_funcs=None, kernel_args_list=None,
                 image_size=32, fit_bias=True, alpha=0.0, maxiter=200,
                 # classifier
                 classifier_cls=None,
                 classifier_kwargs=None,
                 # HOG options
                 use_hog=False,
                 hog_cell_size=(4, 4),
                 hog_block_size=(2, 2),
                 hog_block_stride=(1, 1),
                 hog_n_bins=9,
                 **kwargs):
        self.features = CKNSequential(
            in_channels, out_channels_list, kernel_sizes, subsamplings,
            kernel_funcs, kernel_args_list, **kwargs)

        out_ch = out_channels_list[-1]
        spatial = image_size
        for s in subsamplings:
            spatial = math.ceil(spatial / s)
        self.ckn_out_features = spatial * spatial * out_ch

        # HOG feature extractor
        self.use_hog = use_hog
        self.image_size = image_size
        if use_hog:
            self.hog = HOG(
                cell_size=hog_cell_size,
                block_size=hog_block_size,
                block_stride=hog_block_stride,
                n_bins=hog_n_bins,
            )
            self.hog_out_features = self.hog.descriptor_length((image_size, image_size))
            print(f"HOG enabled — descriptor length: {self.hog_out_features}")
        else:
            self.hog = None
            self.hog_out_features = 0

        self.out_features = self.ckn_out_features + self.hog_out_features

        # Classifier — defaults to LinearSVM; swap in any compatible class
        if classifier_cls is None:
            classifier_cls = LinearSVM
        extra = classifier_kwargs or {}
        self.nclass = nclass
        self.classifier = classifier_cls(
            self.out_features, nclass,
            alpha=alpha, fit_bias=fit_bias, maxiter=maxiter,
            **extra,
        )

    def load_ckn_weights(self, weights):
        """Load pre-trained CKN layer weights (list of arrays, one per layer)."""
        for layer, w in zip(self.features.ckn_layers, weights):
            layer.weight.values = w.copy()

    def _hog_representation(self, x):
        """
        Extract HOG features from a batch of NCHW images.
        Returns (N, D_hog).
        """
        imgs_hwc  = x.transpose(0, 2, 3, 1)
        imgs_gray = imgs_hwc.mean(axis=-1)
        imgs_gray = imgs_gray - imgs_gray.min(axis=(1, 2), keepdims=True)
        return self.hog.batch_forward(imgs_gray)

    def representation(self, x):
        """CKN features, optionally concatenated with HOG. Returns (N, D)."""
        ckn_feats = self.features(x).reshape(x.shape[0], -1)
        if not self.use_hog:
            return ckn_feats
        hog_feats = self._hog_representation(x)
        return np.concatenate([ckn_feats, hog_feats], axis=1)

    def forward(self, x):
        return self.classifier(self.representation(x))

    def __call__(self, x):
        return self.forward(x)

    def normalize(self):
        self.features.normalize()

    def unsup_train_ckn(self, data_loader, n_sampling_patches=150000):
        self.features.unsup_train_(data_loader, n_sampling_patches)

    def unsup_train_classifier(self, data_loader):
        X_enc, Y_enc = self._encode(data_loader)
        self.classifier.fit(X_enc, Y_enc)

    def _encode(self, data_loader):
        X_list, Y_list = [], []
        for data, target in data_loader:
            X_list.append(self.representation(data))
            Y_list.append(target)
        return np.concatenate(X_list), np.concatenate(Y_list)

    def get_parameters(self):
        params = []
        for layer in self.features.ckn_layers:
            params.extend(layer.parameters())
        params.extend(self.classifier.parameters())
        return params


# ── Architectures ─────────────────────────────────────────────────────────────

class CKN2(CKNet):
    """2 layers: [32, 32], filters [3, 3], subsampling [2, 6]"""
    def __init__(self, alpha=0.0, maxiter=5000, **kw):
        super().__init__(10, 3, [32, 32], [3, 3], [2, 6],
                         kernel_funcs=['exp', 'exp'],
                         kernel_args_list=[1 / np.sqrt(3), 1 / np.sqrt(3)],
                         fit_bias=True, alpha=alpha, maxiter=maxiter, **kw)


class CKN3(CKNet):
    """3 layers: [64, 128, 128], filters [3, 3, 3], subsampling [2, 2, 1]"""
    def __init__(self, alpha=0.0, maxiter=5000, **kw):
        super().__init__(10, 3, [64, 128, 128], [3, 3, 3], [2, 2, 1],
                         kernel_funcs=['exp', 'exp', 'exp'],
                         kernel_args_list=[0.6, 0.6, 0.6],
                         fit_bias=True, alpha=alpha, maxiter=maxiter, **kw)


class CKN5(CKNet):
    """5 layers: [64,32,64,32,64], filters [3,1,3,1,3], subsampling [2,1,2,1,3]"""
    def __init__(self, alpha=0.0, maxiter=5000, **kw):
        super().__init__(10, 3, [64, 32, 64, 32, 64], [3, 1, 3, 1, 3], [2, 1, 2, 1, 3],
                         kernel_funcs=['exp', 'poly', 'exp', 'poly', 'exp'],
                         kernel_args_list=[0.5, 2, 0.5, 2, 0.5],
                         fit_bias=True, alpha=alpha, maxiter=maxiter, **kw)


class CKN3MultiScale(CKNet):
    """Multi-scale 3-layer CKN: [96, 128, 192], filters [5, 3, 3], subsampling [2, 2, 1]"""
    def __init__(self, alpha=0.0, maxiter=5000, **kw):
        super().__init__(
            10, 3,
            [96, 128, 192],
            [5, 3, 3],
            [2, 2, 1],
            kernel_funcs=['exp', 'exp', 'exp'],
            kernel_args_list=[0.5, 0.6, 0.8],
            fit_bias=True,
            alpha=alpha,
            maxiter=maxiter,
            **kw
        )


MODELS = {'ckn2': CKN2, 'ckn3': CKN3, 'ckn5': CKN5, 'ckn3ms': CKN3MultiScale}
