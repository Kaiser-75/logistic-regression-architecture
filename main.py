from train import LogisticRegression
from preprocessing import preprocess_image
import numpy as np
import os
import time
import matplotlib.pyplot as plt

def load_data_with_preprocessing(folder_path):
    X, y = [], []
    for label in ['0', '1']:
        class_path = os.path.join(folder_path, label)
        for file in os.listdir(class_path):
            if file.lower().endswith('.png'):
                full_path = os.path.join(class_path, file)
                processed = preprocess_image(full_path)
                X.append(processed)
                y.append(int(label))
    return np.array(X), np.array(y)

def split_train_val_test(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

def compute_confusion_matrix(y_true, y_pred, num_classes=2):
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
        ax.text(j, i, str(val), ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data_with_preprocessing("C:\\Users\\mdmunna\\OneDrive - Texas Tech University\\Not sure about items\\Documents\\ML\\Celegans_ModelGen")

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)

    model = LogisticRegression(
        learning_rate=0.01,
        n_iterations=2000,
        l2_lambda=0.15,
        batch_size=32,
        momentum=0.9 
    )

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

   

    start_val = time.time()
    val_preds = model.predict(X_val)
    val_time = time.time() - start_val
    val_acc = np.mean(val_preds == y_val)
    val_cm = compute_confusion_matrix(y_val, val_preds)

    start_test = time.time()
    test_preds = model.predict(X_test)
    test_time = time.time() - start_test
    test_acc = np.mean(test_preds == y_test)
    test_cm = compute_confusion_matrix(y_test, test_preds)

    print(f"\nTraining Time: {train_time:.2f}s")
    print(f"Validation Accuracy: {val_acc:.4f} (Time: {val_time:.2f}s)")
    print("Confusion Matrix (Validation):")
    print(val_cm)

    print(f"\nTest Accuracy: {test_acc:.4f} (Time: {test_time:.2f}s)")
    print("Confusion Matrix (Test):")
    print(test_cm)

    plot_confusion_matrix(val_cm, title="Validation Confusion Matrix")
    plot_confusion_matrix(test_cm, title="Test Confusion Matrix")

if __name__ == "__main__":
    main()
