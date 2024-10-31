import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(48, 48)):
    images, labels = [], []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels

def augment_data(images, labels):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    return datagen.flow(images, labels, batch_size=32)

if __name__ == "__main__":
    data_dir = "data/train"
    images, labels = load_data(data_dir)
    data_gen = augment_data(images, labels)
