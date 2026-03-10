import numpy as np
import cvxopt

class SVM:
    """
    Standard Binary Support Vector Machine using CVXOPT.
    Expects labels y in {-1, 1}.
    """
    def __init__(self, kernel, C=1.0, tol=1e-5):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.alpha = None
        self.b = 0.0
        self.sv_X = None
        self.sv_y = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Compute the kernel matrix
        K = self.kernel(X, X)
        
        # Formulate the Quadratic Programming (QP) problem
        P = cvxopt.matrix(np.outer(y, y) * K, tc='d')
        q = cvxopt.matrix(-np.ones(n_samples), tc='d')
        
        # G alpha <= h (0 <= alpha <= C)
        G = cvxopt.matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))), tc='d')
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)), tc='d')
        
        # A alpha = b (y^T * alpha = 0)
        A = cvxopt.matrix(y.astype(float).reshape(1, -1), tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        
        # Solve the QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])
        
        # Identify support vectors
        sv_indices = alphas > self.tol
        self.alpha = alphas[sv_indices]
        self.sv_X = X[sv_indices]
        self.sv_y = y[sv_indices]
        
        # Calculate intercept/bias (b)
        if len(self.alpha) > 0:
            sv_K = self.kernel(self.sv_X, self.sv_X)
            self.b = np.mean(self.sv_y - np.sum(self.alpha * self.sv_y * sv_K, axis=1))
        else:
            self.b = 0.0

    def decision_function(self, X):
        """Returns the raw margin scores."""
        if self.alpha is None or len(self.alpha) == 0:
            return np.zeros(X.shape[0])
            
        K = self.kernel(X, self.sv_X)
        return np.dot(K, self.alpha * self.sv_y) + self.b

    def predict(self, X):
        """Returns class labels -1 or 1."""
        return np.sign(self.decision_function(X))


class MulticlassSVM:
    """
    Multiclass SVM using the One-vs-Rest (OvR) strategy.
    Wraps the binary SVM class.
    """
    def __init__(self, kernel, C=1.0):
        self.kernel = kernel
        self.C = C
        self.classes_ = None
        self.classifiers = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers = []
        
        for cls in self.classes_:
            # Create binary labels: +1 for the current class, -1 for all others
            binary_y = np.where(y == cls, 1, -1)
            
            # Initialize and train the binary SVM
            svm = SVM(kernel=self.kernel, C=self.C)
            svm.fit(X, binary_y)
            self.classifiers.append(svm)

    def predict(self, X):
        # Gather confidence scores from all binary classifiers
        scores = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, svm in enumerate(self.classifiers):
            scores[:, i] = svm.decision_function(X)
            
        # Pick the class with the highest confidence score
        best_class_indices = np.argmax(scores, axis=1)
        return self.classes_[best_class_indices]