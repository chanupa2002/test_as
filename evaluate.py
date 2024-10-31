import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_data
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model_path="emotion_model.h5", data_dir="data/val"):
    model = load_model(model_path)
    images, labels = load_data(data_dir)
    predictions = np.argmax(model.predict(images), axis=-1)
    print(classification_report(labels, predictions))
    print(confusion_matrix(labels, predictions))

if __name__ == "__main__":
    evaluate_model()
