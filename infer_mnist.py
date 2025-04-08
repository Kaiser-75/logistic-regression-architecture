import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from train import LogisticRegression
import argparse

def preprocess_tif_image(path, size=(28, 28)):
    img = Image.open(path).convert('L')
    img = img.resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np.flatten()

def infer_and_export(model_path, input_dir, output_path="mnist_predictions.xlsx"):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    results = []
    label_counts = {}

    for file in sorted(os.listdir(input_dir)):
        if file.lower().endswith('.tiff'):
            full_path = os.path.join(input_dir, file)
            processed = preprocess_tif_image(full_path).reshape(1, -1)
            prediction = int(model.predict(processed)[0])
            results.append([file, prediction])
            label_counts[prediction] = label_counts.get(prediction, 0) + 1

    df = pd.DataFrame(results, columns=["Filename", "Label"])
    df.loc[len(df.index)] = ["---", "---"]
    for label, count in sorted(label_counts.items()):
        df.loc[len(df.index)] = [f"Total_{label}", count]

    df.to_excel(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    for label, count in label_counts.items():
        print(f"Total {label}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST inference on test .tiff images")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pkl model")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .tif test images")
    parser.add_argument("--output", type=str, default="mnist_predictions.xlsx", help="Excel output filename")

    args = parser.parse_args()
    infer_and_export(args.model, args.input_dir, args.output)
# python infer_mnist.py --model model_mnist.pkl --input_dir ./MNIST_Test --output mnist_predictions.xlsx
