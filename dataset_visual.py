import numpy as np
import matplotlib.pyplot as plt
import os
from preprocessing import preprocess_image

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

def show_sample_images(X, y, num_samples=5, image_shape=(30, 30)):
    plt.figure(figsize=(10, 2.5))
    indices_0 = np.where(y == 0)[0][:num_samples]
    indices_1 = np.where(y == 1)[0][:num_samples]

    for i, idx in enumerate(indices_0):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X[idx].reshape(image_shape), cmap='gray')
        plt.title("Label: 0")
        plt.axis('off')

    for i, idx in enumerate(indices_1):
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(X[idx].reshape(image_shape), cmap='gray')
        plt.title("Label: 1")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    folder_path = "C:\\Users\\mdmunna\\OneDrive - Texas Tech University\\Not sure about items\\Documents\\ML\\Celegans_ModelGen"
    X, y = load_data_with_preprocessing(folder_path)
    show_sample_images(X, y, num_samples=5)

if __name__ == "__main__":
    main()
