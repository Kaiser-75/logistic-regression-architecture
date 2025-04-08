from train import LogisticRegression
from mnist_dataloader import MNISTLoader
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center', fontsize=8)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def main():
    loader = MNISTLoader(
        train_img_path="C:\\Users\\mdmunna\\OneDrive - Texas Tech University\\Not sure about items\\Documents\\ML\\MNIST\\MNIST\\train-images.idx3-ubyte",
        train_lbl_path="C:\\Users\\mdmunna\\OneDrive - Texas Tech University\\Not sure about items\\Documents\\ML\\MNIST\\MNIST\\train-labels.idx1-ubyte",
        test_img_path="C:\\Users\\mdmunna\\OneDrive - Texas Tech University\\Not sure about items\\Documents\\ML\\MNIST\\MNIST\\t10k-images.idx3-ubyte",
        test_lbl_path="C:\\Users\\mdmunna\\OneDrive - Texas Tech University\\Not sure about items\\Documents\\ML\\MNIST\\MNIST\\t10k-labels.idx1-ubyte"
    )

    X_train, y_train, X_test, y_test = loader.load_data()

    model = LogisticRegression(
        learning_rate=0.0025,
        n_iterations=100,
        batch_size=32,
        momentum=0.9,
        l2_lambda=0.002
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    test_cm = compute_confusion_matrix(y_test, y_pred)

    print(f"\nTraining Time: {train_time:.2f}s")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)

    plot_confusion_matrix(test_cm, title="MNIST Test Confusion Matrix")

    with open("model_mnist.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to model_mnist.pkl")

if __name__ == "__main__":
    main()
