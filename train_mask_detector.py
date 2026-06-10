import os
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np

# ────────────────────────────────────────────────
# BUILD MODEL
# ────────────────────────────────────────────────
def build_model():
    baseModel = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(128, 128, 3))
    )

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten()(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    return model

# ────────────────────────────────────────────────
# LOAD DATASET
# ────────────────────────────────────────────────
def load_dataset(dataset_path):
    data, labels = [], []
    categories = ["with_mask", "without_mask"]

    for category in categories:
        path = os.path.join(dataset_path, category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = load_img(img_path, target_size=(128, 128))
                image = img_to_array(image) / 255.0
                data.append(image)
                labels.append(category)
            except:
                pass

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    return np.array(data), np.array(labels)

# ────────────────────────────────────────────────
# TRAIN MULTIPLE DATASETS & EPOCHS
# ────────────────────────────────────────────────
def train_multiple_datasets(datasets):
    EPOCH_LIST = [10, 20, 30]
    BS = 8
    INIT_LR = 1e-4

    os.makedirs("comparison_results", exist_ok=True)

    results = {}

    for dataset in datasets:
        print(f"\n[INFO] Dataset: {dataset}")
        results[dataset] = {}

        data, labels = load_dataset(dataset)
        (trainX, testX, trainY, testY) = train_test_split(
            data, labels, test_size=0.2, stratify=labels
        )

        aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        for ep in EPOCH_LIST:
            print(f"[INFO] Training for {ep} epochs")

            model = build_model()
            opt = Adam(learning_rate=INIT_LR)
            model.compile(
                loss="binary_crossentropy",
                optimizer=opt,
                metrics=["accuracy"]
            )

            H = model.fit(
                aug.flow(trainX, trainY, batch_size=BS),
                steps_per_epoch=len(trainX) // BS,
                validation_data=(testX, testY),
                epochs=ep,
                verbose=1
            )

            results[dataset][ep] = {
                "accuracy": H.history["accuracy"][-1],
                "val_accuracy": H.history["val_accuracy"][-1]
            }

    return results

# ────────────────────────────────────────────────
# PLOTTING
# ────────────────────────────────────────────────
def plot_results(results):
    epochs = [10, 20, 30]

    # Dataset-wise graphs
    for dataset in results:
        acc = [results[dataset][e]["accuracy"] for e in epochs]
        plt.plot(epochs, acc, marker="o")
        plt.title(f"{dataset} Accuracy vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.savefig(f"comparison_results/{dataset}_comparison.png")
        plt.show()

    # Combined graph
    for dataset in results:
        acc = [results[dataset][e]["accuracy"] for e in epochs]
        plt.plot(epochs, acc, marker="o", label=dataset)

    plt.title("All Datasets & Epochs Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("comparison_results/all_datasets_comparison.png")
    plt.show()

# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    results = train_multiple_datasets(args.datasets)
    plot_results(results)
