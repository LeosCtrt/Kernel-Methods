import numpy as np
from scipy.ndimage import convolve1d


class Module:
    """Base class mirroring torch.nn.Module."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        extra = self.extra_repr()
        lines = [f"{self.__class__.__name__}({extra})"]
        for name, child in self._named_direct_children():
            indented = "\n".join("  " + l for l in repr(child).splitlines())
            lines.append(f"  ({name}): {indented.lstrip()}")
        if len(lines) > 1:
            return lines[0][:-1] + "\n" + "\n".join(lines[1:]) + "\n)"
        return lines[0]

    def _named_direct_children(self):
        return [(k, v) for k, v in self.__dict__.items() if isinstance(v, Module)]


class Sequential(Module):
    """Chain of Modules applied one after another."""

    def __init__(self, *modules: Module):
        self._modules = [(str(i), m) for i, m in enumerate(modules)]

    def forward(self, x):
        for _, module in self._modules:
            x = module(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return self._modules[idx][1]

    def __len__(self) -> int:
        return len(self._modules)

    def _named_direct_children(self):
        return self._modules

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules:
            indented = "\n".join("  " + l for l in repr(module).splitlines())
            lines.append(f"  ({name}): {indented.lstrip()}")
        lines.append(")")
        return "\n".join(lines)


class GammaCorrection(Module):
    """Square-root gamma normalisation (Dalal & Triggs §2.1)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def extra_repr(self):
        return f"enabled={self.enabled}"

    def forward(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=float)
        if not self.enabled:
            return image
        m = image.max()
        return np.sqrt(image / m) if m > 0 else image


class GradientComputation(Module):
    """
    Centred [-1, 0, 1] gradients; for colour images keeps the
    channel with the highest per-pixel magnitude (Dalal & Triggs §2.2).
    Returns (magnitude, orientation) both (H, W).
    """

    _KERNEL = np.array([-1.0, 0.0, 1.0])

    def extra_repr(self):
        return "kernel=[-1, 0, 1], orientation=unsigned"

    def forward(self, image: np.ndarray):
        channels = (
            [image] if image.ndim == 2
            else [image[:, :, c] for c in range(image.shape[2])]
        )
        best_mag = best_gx = best_gy = None
        for ch in channels:
            gx = convolve1d(ch, self._KERNEL, axis=1, mode="constant", cval=0.0)
            gy = convolve1d(ch, self._KERNEL, axis=0, mode="constant", cval=0.0)
            mag = np.hypot(gx, gy)
            if best_mag is None or mag.sum() > best_mag.sum():
                best_mag, best_gx, best_gy = mag, gx, gy
        orientation = np.rad2deg(np.arctan2(best_gy, best_gx)) % 180.0
        return best_mag, orientation


class CellHistograms(Module):
    """
    Soft-vote gradient magnitudes into per-cell orientation histograms
    with bilinear bin interpolation (Dalal & Triggs §2.3).
    """

    def __init__(self, cell_size=(8, 8), n_bins=9):
        self.cell_size = cell_size
        self.n_bins = n_bins

    def extra_repr(self):
        return f"cell_size={self.cell_size}, n_bins={self.n_bins}"

    def forward(self, inputs):
        magnitude, orientation = inputs
        H, W = magnitude.shape
        cy, cx = self.cell_size
        n_cells_y = H // cy
        n_cells_x = W // cx
        bin_width = 180.0 / self.n_bins

        hist = np.zeros((n_cells_y, n_cells_x, self.n_bins), dtype=float)
        ys = np.arange(n_cells_y * cy)
        xs = np.arange(n_cells_x * cx)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        mag = magnitude[:n_cells_y * cy, :n_cells_x * cx]
        ori = orientation[:n_cells_y * cy, :n_cells_x * cx]

        bin_float = ori / bin_width - 0.5
        bin0 = np.floor(bin_float).astype(int) % self.n_bins
        bin1 = (bin0 + 1) % self.n_bins
        weight1 = bin_float - np.floor(bin_float)
        weight0 = 1.0 - weight1

        flat_cell = (yy // cy) * n_cells_x + (xx // cx)
        hist_flat = hist.reshape(-1, self.n_bins)
        np.add.at(hist_flat, (flat_cell.ravel(), bin0.ravel()), (mag * weight0).ravel())
        np.add.at(hist_flat, (flat_cell.ravel(), bin1.ravel()), (mag * weight1).ravel())
        return hist


class BlockNormalization(Module):
    """
    L2-Hys block normalisation over overlapping cell blocks
    (Dalal & Triggs §2.4): L2 → clip → L2.
    """

    def __init__(self, block_size=(2, 2), block_stride=(1, 1), eps=1e-5, clip=0.2):
        self.block_size = block_size
        self.block_stride = block_stride
        self.eps = eps
        self.clip = clip

    def extra_repr(self):
        return (f"block_size={self.block_size}, "
                f"block_stride={self.block_stride}, clip={self.clip}")

    def forward(self, hist: np.ndarray) -> np.ndarray:
        n_cells_y, n_cells_x, n_bins = hist.shape
        by, bx = self.block_size
        sy, sx = self.block_stride
        n_blocks_y = (n_cells_y - by) // sy + 1
        n_blocks_x = (n_cells_x - bx) // sx + 1
        block_len = by * bx * n_bins
        descriptor = np.empty(n_blocks_y * n_blocks_x * block_len, dtype=float)
        ptr = 0
        for iy in range(n_blocks_y):
            for ix in range(n_blocks_x):
                block = hist[iy * sy:iy * sy + by, ix * sx:ix * sx + bx, :].ravel()
                block = block / np.sqrt(np.dot(block, block) + self.eps ** 2)
                block = np.clip(block, 0.0, self.clip)
                block = block / np.sqrt(np.dot(block, block) + self.eps ** 2)
                descriptor[ptr:ptr + block_len] = block
                ptr += block_len
        return descriptor


class HOG(Sequential):
    """
    Full HOG pipeline: GammaCorrection → GradientComputation
                       → CellHistograms → BlockNormalization.

    Accepts a single (H, W) or (H, W, C) image and returns a 1-D descriptor.
    Use batch_forward() for a batch of images.
    """

    def __init__(self, cell_size=(8, 8), block_size=(2, 2),
                 block_stride=(1, 1), n_bins=9, gamma_correction=True):
        super().__init__(
            GammaCorrection(enabled=gamma_correction),
            GradientComputation(),
            CellHistograms(cell_size=cell_size, n_bins=n_bins),
            BlockNormalization(block_size=block_size, block_stride=block_stride),
        )
        self.gamma = self[0]
        self.gradient = self[1]
        self.cell_hists = self[2]
        self.block_norm = self[3]
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.n_bins = n_bins

    def descriptor_length(self, image_size):
        """Compute output dimensionality without running the pipeline."""
        H, W = image_size
        cy, cx = self.cell_size
        by, bx = self.block_size
        sy, sx = self.block_stride
        n_blocks_y = (H // cy - by) // sy + 1
        n_blocks_x = (W // cx - bx) // sx + 1
        return n_blocks_y * n_blocks_x * by * bx * self.n_bins

    def batch_forward(self, images: np.ndarray) -> np.ndarray:
        """
        Process a batch of images.

        Parameters
        ----------
        images : (N, H, W) or (N, H, W, C)  — HWC layout

        Returns
        -------
        descriptors : (N, D)
        """
        return np.stack([self.forward(img) for img in images])
