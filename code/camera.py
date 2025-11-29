import cv2
from PIL import Image
import numpy as np
import joblib

# Import your embedding model + function
from train_one_class_model import load_embedding_model, extract_embedding

# Load models
print("Loading models...")
svm = joblib.load("one_class_model.pkl")
embedding_model = load_embedding_model()

def classify_frame(frame_path):
    # # Convert OpenCV BGR → RGB
    # frame = cv2.imread(frame_path)
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    # # Convert NumPy array → PIL Image
    # pil_img = Image.fromarray(rgb)
    #
    # # Extract embedding
    emb = extract_embedding(frame_path, embedding_model).reshape(1, -1)

    # Predict
    label = svm.predict(emb)[0]
    score = svm.decision_function(emb)[0]

    return label, score

# Open webcam (0 is default laptop camera)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Could not open camera.")
    exit()

print("Press SPACE to classify. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live video feed
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # spacebar
        print("Capturing frame...")
        cv2.imwrite("../images/test/cleaned/test4.jpg", frame)
        label, score = classify_frame('../images/test/cleaned/test4.jpg')

        if label == 1:
            print(f"IN category (score: {score:.3f})")
        else:
            print(f"NOT in category (score: {score:.3f})")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
