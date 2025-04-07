import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, momentum=0.0, l2_lambda=0.0, batch_size=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.weights = None  
        self.bias = None     

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)  
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y, n_classes):
        return np.eye(n_classes)[y]

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        base_loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        l2_penalty = (self.l2_lambda / 2) * np.sum(self.weights ** 2)
        return base_loss + l2_penalty

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))  
        y_onehot = self.one_hot(y, n_classes)

        v_w = np.zeros_like(self.weights)
        v_b = np.zeros_like(self.bias)

        for epoch in range(self.n_iterations):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]

                logits = np.dot(X_batch, self.weights) + self.bias
                probs = self.softmax(logits)
                error = probs - y_batch

                grad_weights = np.dot(X_batch.T, error) / len(X_batch) + self.l2_lambda * self.weights
                grad_bias = np.sum(error, axis=0, keepdims=True) / len(X_batch)

                v_w = self.momentum * v_w + self.learning_rate * grad_weights
                v_b = self.momentum * v_b + self.learning_rate * grad_bias

                self.weights -= v_w
                self.bias -= v_b

            if epoch % 10 == 0:
                logits_all = np.dot(X, self.weights) + self.bias
                probs_all = self.softmax(logits_all)
                loss = self.cross_entropy_loss(y_onehot, probs_all)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Training Accuracy: {accuracy:.4f}")

    def predict_proba(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
