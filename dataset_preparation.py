# step1_dataset_preparation.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_images(base_path, img_size=(224, 224)):
    data = []
    labels = []
    for label in ['genuine', 'forged']:
        path = os.path.join(base_path, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img.astype('float32') / 255.0
                data.append(np.expand_dims(img, axis=-1))
                labels.append(0 if label == 'genuine' else 1)
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    data_path = 'datasets/CEDAR/'  # change to target dataset
    X, y = load_and_preprocess_images(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    np.savez_compressed('step1_dataset_prepared.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
