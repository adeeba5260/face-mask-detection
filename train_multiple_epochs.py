"""
Train mask detector on MULTIPLE DATASETS
For each dataset → epochs [10,20,30]
Creates:
- Dataset-wise comparison graphs
- FINAL combined graph (all datasets + all epochs)
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json

IMG_SIZE = (128, 128)
EPOCH_LIST = [10, 20, 30]
BATCH_SIZE = 8
INIT_LR = 1e-4


# ────────────────────────────────────────────────
# LOAD DATASET
# ────────────────────────────────────────────────
def load_data(dataset_path):
    data, labels = [], []

    for imagePath in paths.list_images(dataset_path):
        label = imagePath.split(os.path.sep)[-2]
        image = load_img(imagePath, target_size=IMG_SIZE)
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = to_categorical(lb.fit_transform(labels))

    return train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )


# ────────────────────────────────────────────────
# BUILD MODEL
# ────────────────────────────────────────────────
def build_model():
    baseModel = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(128, 128, 3))
    )

    head = baseModel.output
    head = AveragePooling2D(pool_size=(4, 4))(head)
    head = Flatten()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(2, activation="softmax")(head)

    model = Model(inputs=baseModel.input, outputs=head)

    for layer in baseModel.layers:
        layer.trainable = False

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=INIT_LR),
        metrics=["accuracy"]
    )

    return model


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    os.makedirs("comparison_results", exist_ok=True)
    results = {}

    for dataset in args.datasets:
        print(f"\n[INFO] Dataset: {dataset}")
        results[dataset] = {}

        trainX, testX, trainY, testY = load_data(dataset)

        for EPOCHS in EPOCH_LIST:
            print(f"[INFO] Training for {EPOCHS} epochs")

            model = build_model()

            H = model.fit(
                aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                steps_per_epoch=len(trainX) // BATCH_SIZE,
                validation_data=(testX, testY),
                epochs=EPOCHS,
                verbose=1
            )

            results[dataset][EPOCHS] = H.history["val_accuracy"][-1]

            model.save(f"comparison_results/{dataset}_{EPOCHS}.keras")

    # ────────────────────────────────────────────────
    # DATASET-WISE GRAPHS
    # ────────────────────────────────────────────────
    for dataset in results:
        acc = [results[dataset][e] for e in EPOCH_LIST]
        plt.plot(EPOCH_LIST, acc, marker="o")
        plt.title(f"{dataset} Accuracy vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.grid()
        plt.savefig(f"comparison_results/{dataset}_comparison.png")
        plt.show()

    # ────────────────────────────────────────────────
    # FINAL COMBINED GRAPH
    # ────────────────────────────────────────────────
    for dataset in results:
        acc = [results[dataset][e] for e in EPOCH_LIST]
        plt.plot(EPOCH_LIST, acc, marker="o", label=dataset)

    plt.title("All Datasets & Epochs Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("comparison_results/all_datasets_comparison.png")
    plt.show()

    with open("comparison_results/final_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("[INFO] All experiments completed successfully.")


if __name__ == "__main__":
    main()
