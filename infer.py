import os
import numpy as np
import pandas as pd
import pickle
from preprocessing import preprocess_image
from train import LogisticRegression
import argparse

def infer_and_export(model_path, input_dir, output_path="predictions.xlsx"):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    results = []
    label_counts = {0: 0, 1: 0}

    for file in sorted(os.listdir(input_dir)):
        if file.lower().endswith(('.png', '.tif')):
            full_path = os.path.join(input_dir, file)
            processed = preprocess_image(full_path).reshape(1, -1)
            prediction = int(model.predict(processed)[0])
            results.append([file, prediction])
            label_counts[prediction] += 1

    df = pd.DataFrame(results, columns=["Filename", "Label"])
    df.loc[len(df.index)] = ["Total_0", label_counts[0]]
    df.loc[len(df.index)] = ["Total_1", label_counts[1]]
    df.to_excel(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Total 0: {label_counts[0]}, Total 1: {label_counts[1]}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference with trained model and test images.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pkl)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing test images (*.png)")
    parser.add_argument("--output", type=str, default="predictions.xlsx", help="Output Excel filename")

    args = parser.parse_args()

    infer_and_export(args.model, args.input_dir, args.output)
###------------------------------------####
### Use below comand to run the inference script from the command line:
# python infer.py --model model_celegans.pkl --input_dir ./Testcelegans --output worm_predictions.xlsx  #
    
