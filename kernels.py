import numpy as np

class Kernel:
    """
    Base class for all kernels. 
    Every kernel should implement the __call__ method to compute the 
    kernel matrix between two datasets X and Y.
    """
    def __call__(self, X, Y=None):
        raise NotImplementedError("Kernel __call__ method must be implemented by subclasses.")

class LinearKernel(Kernel):
    """
    Linear Kernel.
    """
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        return np.dot(X, Y.T)

class PolynomialKernel(Kernel):
    """
    Polynomial Kernel.
    """
    def __init__(self, degree=3, gamma=None, coef0=1.0):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        
        gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
        return (gamma * np.dot(X, Y.T) + self.coef0) ** self.degree

class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) / Gaussian Kernel.
    """
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
            
        gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
        
        # Efficient squared Euclidean distance computation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2(x . y)
        X_norm = np.sum(X ** 2, axis=-1)[:, np.newaxis]
        Y_norm = np.sum(Y ** 2, axis=-1)[np.newaxis, :]
        
        # Calculate squared distances and ensure no negative distances due to floating point inaccuracies
        sq_dists = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        sq_dists = np.maximum(sq_dists, 0.0) 
        
        return np.exp(-gamma * sq_dists)

class LaplacianKernel(Kernel):
    """
    Laplacian Kernel.
    """
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
            
        gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
        
        # Using numpy broadcasting to compute L1 distances (Manhattan distance)
        # Note: This can be memory intensive for very large matrices.
        diffs = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        l1_dists = np.sum(np.abs(diffs), axis=-1)
        
        return np.exp(-gamma * l1_dists)

class SigmoidKernel(Kernel):
    """
    Sigmoid Kernel.
    """
    def __init__(self, gamma=None, coef0=0.0):
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
            
        gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
        return np.tanh(gamma * np.dot(X, Y.T) + self.coef0)

class CosineKernel(Kernel):
    """
    Cosine Similarity Kernel.
    """
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
            
        # Normalize rows to unit vectors
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        
        return np.dot(X_norm, Y_norm.T)