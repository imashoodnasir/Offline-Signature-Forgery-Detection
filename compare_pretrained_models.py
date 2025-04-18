# compare_pretrained_models.py

from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, DenseNet121, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = np.load('step1_dataset_prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Resize to 224x224x3 for pretrained models expecting RGB input
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

# Base models dictionary
pretrained_models = {
    'VGG16': VGG16,
    'ResNet50': ResNet50,
    'MobileNetV2': MobileNetV2,
    'DenseNet121': DenseNet121,
    'Xception': Xception
}

def build_transfer_model(base_model_fn, name):
    base_model = base_model_fn(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=outputs, name=f"{name}_Transfer")
    for layer in base_model.layers:
        layer.trainable = False
    return model

# Train and evaluate
def train_and_evaluate_transfer(model, name):
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train_rgb, y_train, validation_split=0.1, epochs=10, batch_size=32,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=0)

    y_pred = (model.predict(X_test_rgb) > 0.5).astype('int32')
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with open("pretrained_comparison_results.txt", "a") as f:
        f.write(f"{name}:\n")
        f.write(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}\n\n")

if __name__ == "__main__":
    for name, model_fn in pretrained_models.items():
        model = build_transfer_model(model_fn, name)
        train_and_evaluate_transfer(model, name)
