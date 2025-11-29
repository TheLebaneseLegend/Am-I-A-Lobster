import os
import glob
import numpy as np
from PIL import Image
import joblib

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.svm import OneClassSVM


# -----------------------------
# 1. Load Pretrained Model
# -----------------------------
def load_embedding_model():
    """
    Loads a pretrained ResNet50 and removes its final classification layer
    so it outputs a 2048-dimensional embedding.
    """
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()  # remove classification head
    model.eval()
    return model


# -----------------------------
# 2. Define Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# 3. Extract Embedding for One Image
# -----------------------------
def extract_embedding(img_path, model):
    """
    Loads an image, preprocesses it, and returns its 2048-d embedding.
    """
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        embedding = model(x).squeeze().numpy()

    return embedding


# -----------------------------
# 4. Batch Process All Images in a Folder
# -----------------------------
def extract_embeddings_from_folder(folder, model):
    """
    Extracts embeddings for every .jpg or .jpeg in a folder.
    """
    image_paths = glob.glob(os.path.join(folder, "*.jpg")) + \
                  glob.glob(os.path.join(folder, "*.jpeg"))

    embeddings = []

    for img_path in image_paths:
        emb = extract_embedding(img_path, model)
        embeddings.append(emb)
        print(f"Processed {img_path}")

    embeddings = np.vstack(embeddings)
    return embeddings, image_paths


# -----------------------------
# 5. Train One-Class SVM
# -----------------------------
def train_one_class_model(embeddings):
    """
    Trains a one-class SVM bubble.
    """
    ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
    ocsvm.fit(embeddings)
    return ocsvm


# -----------------------------
# 6. Predict Function for New Images
# -----------------------------
def predict_image(img_path, model, ocsvm):
    """
    Returns +1 if inside category, -1 if anomaly.
    """
    emb = extract_embedding(img_path, model).reshape(1, -1)
    label = ocsvm.predict(emb)[0]
    score = ocsvm.decision_function(emb)[0]
    return label, score


# -----------------------------
# 7. Main Training Pipeline
# -----------------------------
def main():
    # >>>>> CHANGE THIS TO THE FOLDER ON YOUR COMPUTER <<<<<
    training_folder = '/Users/cameronconnelly/Documents/Coding Projects/am_i_a_lobster/images/train/cleaned'

    print("Loading model...")
    model = load_embedding_model()

    print("Extracting embeddings...")
    embeddings, image_paths = extract_embeddings_from_folder(training_folder, model)

    print("Training one-class SVM...")
    ocsvm = train_one_class_model(embeddings)

    # save everything
    joblib.dump(ocsvm, "one_class_model.pkl")
    np.save("train_embeddings.npy", embeddings)

    print("\nTraining complete!")
    print(f"Processed {len(image_paths)} images.")
    print("Model saved as one_class_model.pkl")
    print("Embeddings saved as train_embeddings.npy")


if __name__ == "__main__":
    main()
