import numpy as np


class LogisticRegression:
    """
    Logistic Regression implemented from scratch using Gradient Descent.
    No ML libraries used — pure math with NumPy.
    """

    def __init__(self, eta=0.001, epochs=1000):
        """
        Initialize hyperparameters.

        Parameters:
        -----------
        eta    : float — Learning rate (how big each update step is)
        epochs : int   — Number of training iterations
        """
        self.eta = eta
        self.epochs = epochs

    def _sigmoid(self, z):
        """
        Sigmoid activation function.
        Maps any real number to a probability between 0 and 1.

        Formula: σ(z) = 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features) — Training features
        y : numpy array of shape (n_samples,)            — True labels (0 or 1)
        """
        n_samples, n_features = X.shape

        # Initialize weights randomly (small values)
        self.weights = np.random.rand(n_features)

        # Track cost over epochs to monitor learning
        self.cost_history = np.zeros(self.epochs)

        for i in range(self.epochs):

            # Step 1: Linear combination z = X · w
            z = X @ self.weights

            # Step 2: Apply sigmoid to get predicted probabilities
            y_hat = self._sigmoid(z)

            # Step 3: Compute Binary Cross-Entropy cost
            # Small epsilon (1e-10) added to avoid log(0)
            self.cost_history[i] = -np.mean(
                y * np.log(y_hat + 1e-10) + (1 - y) * np.log(1 - y_hat + 1e-10)
            )

            # Step 4: Compute gradient of cost w.r.t weights
            # dJ/dw = (1/n) * X^T · (y_hat - y)
            dw = (1 / n_samples) * ((y_hat - y) @ X)

            # Step 5: Update weights using Gradient Descent
            self.weights -= self.eta * dw

            # Early stopping if cost is very small
            if self.cost_history[i] < 1e-6:
                print(f"Converged at epoch {i + 1}!")
                break

    def predict_proba(self, X):
        """
        Predict probability that each sample belongs to class 1.

        Returns:
        --------
        Array of probabilities between 0 and 1.
        """
        z = X @ self.weights
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1) using a probability threshold.

        Parameters:
        -----------
        threshold : float — Cutoff probability (default 0.5)

        Returns:
        --------
        Array of predicted labels (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        """
        Calculate prediction accuracy on a dataset.

        Formula: accuracy = correct predictions / total predictions

        Returns:
        --------
        Float between 0 and 1 representing accuracy.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)