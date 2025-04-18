# robustness_evaluation.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import array_to_img
import cv2
from sklearn.metrics import accuracy_score

# Load model and data
model = load_model('mslgnet_final_model.h5')
data = np.load('step1_dataset_prepared.npz')
X_test, y_test = data['X_test'], data['y_test']

loss_fn = BinaryCrossentropy()

# FGSM Attack
def fgsm_attack(images, labels, epsilon=0.01):
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)

    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images)
        loss = loss_fn(labels, predictions)

    gradient = tape.gradient(loss, images)
    signed_grad = tf.sign(gradient)
    perturbed_images = images + epsilon * signed_grad
    perturbed_images = tf.clip_by_value(perturbed_images, 0, 1)
    return perturbed_images.numpy()

# PGD Attack
def pgd_attack(images, labels, epsilon=0.01, alpha=0.002, iterations=10):
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    adv_images = tf.identity(images)

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            predictions = model(adv_images)
            loss = loss_fn(labels, predictions)
        gradient = tape.gradient(loss, adv_images)
        adv_images = adv_images + alpha * tf.sign(gradient)
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0, 1)
    return adv_images.numpy()

# JPEG Compression
def jpeg_compression(images, quality=50):
    compressed = []
    for img in images:
        img = np.uint8(img * 255)
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
        compressed.append(np.expand_dims(dec, axis=-1))
    return np.array(compressed)

# Gaussian Noise
def add_gaussian_noise(images, std=0.01):
    noise = np.random.normal(0, std, images.shape)
    noisy = images + noise
    return np.clip(noisy, 0, 1)

# Evaluate robustness
def evaluate_under_attack(perturbed, y_true):
    preds = (model.predict(perturbed) > 0.5).astype('int32')
    acc = accuracy_score(y_true, preds)
    return acc

# Convert labels to one-hot (required by TF loss)
y_test_tf = tf.convert_to_tensor(np.expand_dims(y_test, axis=-1).astype('float32'))

fgsm_imgs = fgsm_attack(X_test, y_test_tf)
pgd_imgs = pgd_attack(X_test, y_test_tf)
jpeg_imgs = jpeg_compression(X_test)
noisy_imgs = add_gaussian_noise(X_test)

acc_fgsm = evaluate_under_attack(fgsm_imgs, y_test)
acc_pgd = evaluate_under_attack(pgd_imgs, y_test)
acc_jpeg = evaluate_under_attack(jpeg_imgs, y_test)
acc_noise = evaluate_under_attack(noisy_imgs, y_test)

# Save results
with open("robustness_results.txt", "w") as f:
    f.write(f"FGSM Accuracy: {acc_fgsm:.4f}\n")
    f.write(f"PGD Accuracy: {acc_pgd:.4f}\n")
    f.write(f"JPEG Compression Accuracy: {acc_jpeg:.4f}\n")
    f.write(f"Gaussian Noise Accuracy: {acc_noise:.4f}\n")
