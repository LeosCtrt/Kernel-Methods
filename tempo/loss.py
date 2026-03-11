import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Loss functions
#  Both expose the same interface:
#    loss = LossClass()(output, target)   →  loss.item  (scalar)
#    dW, db = loss.backward(x)            →  gradients for a linear head
# ─────────────────────────────────────────────────────────────────────────────

class SquaredHingeLoss:
    """
    One-vs-all squared hinge loss (paper §3, §4.1):

        L = (1/N) Σ_i Σ_c  max(0, 1 − y_{ic} · ŷ_{ic})²

    where y_{ic} = +1 if label of i is c, else −1.
    This is the loss used in the paper for CIFAR-10 / SVHN classification.

    Attributes:
        item (float): Scalar loss value after calling the instance.
    """

    def __init__(self):
        self.item = None

    def __call__(self, output: np.ndarray, target: np.ndarray) -> "SquaredHingeLoss":
        """
        Compute squared hinge loss.

        Args:
            output (ndarray): Scores, shape (N, C).
            target (ndarray): Integer class labels, shape (N,).

        Returns:
            self  (allows  loss = criterion(out, tgt);  loss.item)
        """
        N, C       = output.shape
        target     = target.astype(int)
        # Build ±1 label matrix
        Y          = -np.ones((N, C))
        Y[np.arange(N), target] = 1.0

        self.margin     = np.maximum(0.0, 1.0 - Y * output)   # (N, C)
        self.item       = (self.margin ** 2).mean()
        self._Y         = Y
        self._N         = N
        self._C         = C
        return self

    def backward(self, x: np.ndarray):
        """
        Gradients of the loss w.r.t. a linear head  output = x @ Wᵀ + b.

        Args:
            x (ndarray): Representations fed into the linear head, shape (N, D).

        Returns:
            dW (ndarray): shape (C, D)
            db (ndarray): shape (C, 1)
        """
        # d L / d(output)  =  −2 Y ⊙ margin / (N·C)
        dout = -2.0 * self._Y * self.margin / (self._N * self._C)  # (N, C)
        dW   = dout.T @ x                                           # (C, D)
        db   = dout.sum(axis=0, keepdims=True).T                    # (C, 1)
        return dW, db


class CrossEntropyLoss:
    """
    Softmax cross-entropy loss.

    Attributes:
        item (float): Scalar loss value after calling the instance.
    """

    def __init__(self):
        self.item = None

    def __call__(self, output: np.ndarray, target: np.ndarray) -> "CrossEntropyLoss":
        """
        Args:
            output (ndarray): Logits, shape (N, C).
            target (ndarray): Integer class labels, shape (N,).
        """
        target = target.astype(int)
        self.target = target

        exp_s = np.exp(output - output.max(axis=1, keepdims=True))
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)

        self.probabilities = probs
        self.item = -np.mean(np.log(probs[np.arange(len(output)), target] + 1e-15))
        return self

    def backward(self, x: np.ndarray):
        """
        Args:
            x (ndarray): Representations, shape (N, D).

        Returns:
            dW (ndarray): shape (C, D)
            db (ndarray): shape (C, 1)
        """
        grad = self.probabilities.copy()
        grad[np.arange(len(self.probabilities)), self.target] -= 1
        grad /= len(self.probabilities)
        dW = grad.T @ x
        db = grad.sum(axis=0, keepdims=True).T
        return dW, db


LOSS = {
    "sq_hinge": SquaredHingeLoss,
    "ce":       CrossEntropyLoss,
}
