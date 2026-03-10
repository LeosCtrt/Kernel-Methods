import numpy as np

class KernelPCA:
    """
    Kernel Principal Component Analysis (KPCA).
    Non-linear dimensionality reduction using the kernel trick.
    """
    def __init__(self, kernel, n_components=None):
        self.kernel = kernel
        self.n_components = n_components
        self.alphas = None
        self.lambdas = None
        self.X_train = None

    def fit_transform(self, X):
        self.X_train = X
        n_samples = X.shape[0]

        # 1. Compute the Gram matrix
        K = self.kernel(X, X)

        # 2. Center the kernel matrix
        # In feature space, we must mean-center our data. 
        # Kc = K - 1_n K - K 1_n + 1_n K 1_n
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # 3. Eigenvalue decomposition
        # eigh is highly optimized for symmetric matrices like our Gram matrix
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 4. Select the top 'n_components'
        if self.n_components is not None:
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]

        self.lambdas = eigenvalues
        
        # Normalize the eigenvectors
        # alpha_i = alpha_i / sqrt(lambda_i)
        self.alphas = eigenvectors / np.sqrt(self.lambdas + 1e-8)

        # Return the projected training data
        return np.dot(K_centered, self.alphas)

    def transform(self, X):
        """Projects new test data into the fitted KPCA space."""
        if self.X_train is None:
            raise ValueError("KPCA has not been fitted yet.")
            
        n_samples = self.X_train.shape[0]
        
        # Compute kernel between test data and training data
        K_test = self.kernel(X, self.X_train)
        K_train = self.kernel(self.X_train, self.X_train)

        # Center the test kernel matrix using the training data statistics
        one_n = np.ones((n_samples, n_samples)) / n_samples
        one_test = np.ones((X.shape[0], n_samples)) / n_samples

        K_test_centered = (K_test 
                           - one_test.dot(K_train) 
                           - K_test.dot(one_n) 
                           + one_test.dot(K_train).dot(one_n))

        # Project onto the normalized eigenvectors
        return np.dot(K_test_centered, self.alphas)