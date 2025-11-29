import joblib
import numpy as np
from train_one_class_model import load_embedding_model, extract_embedding

ocsvm = joblib.load("one_class_model.pkl")
model = load_embedding_model()

img_path = "../images/test/cleaned/test2_cleaned.jpg"
emb = extract_embedding(img_path, model).reshape(1, -1)

label = ocsvm.predict(emb)[0]
score = ocsvm.decision_function(emb)[0]

print("Label:", "LOBSTER" if label == 1 else "NOT a lobster")
print("Score:", score)
