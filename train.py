import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
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
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m  

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))  
        y_onehot = self.one_hot(y, n_classes)

        for i in range(self.n_iterations):
            logits = np.dot(X, self.weights) + self.bias   
            probs = self.softmax(logits)                   
            if i % 10 == 0:
                loss = self.cross_entropy_loss(y_onehot, probs)
                print(f"Iteration {i}: Loss = {loss:.4f}")
            error = probs - y_onehot                       
            grad_weights = (1 / n_samples) * np.dot(X.T, error)  
            grad_bias = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)  
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Training Accuracy: {accuracy:.4f}")

    def predict_proba(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
