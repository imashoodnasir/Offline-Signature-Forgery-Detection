# ablation_study.py

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Reshape, Multiply, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load prepared data
data = np.load('step1_dataset_prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# --- ABLATION VARIANTS ---

# Variant 1: Without MSFA block
def build_without_msfa(input_shape=(224, 224, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(128, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# Variant 2: Without LGFI block
def build_without_lgfi(input_shape=(224, 224, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# Variant 3: Only One MSFA Block
def build_single_msfa(input_shape=(224, 224, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Simple MSFA block
    gap = GlobalAveragePooling2D()(x)
    dense = Dense(x.shape[-1], activation='sigmoid')(gap)
    attention = Reshape((1, 1, x.shape[-1]))(dense)
    x = Multiply()([x, attention])

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# --- TRAINING FUNCTION ---
def train_and_evaluate(model, name):
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              validation_split=0.1,
              epochs=10,
              batch_size=32,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
              verbose=0)

    preds = (model.predict(X_test) > 0.5).astype('int32')
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    with open("ablation_results.txt", "a") as f:
        f.write(f"{name}:\n")
        f.write(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}\n\n")
    print(f"{name} done. Accuracy: {acc:.4f}")

# --- RUN ALL VARIANTS ---
if __name__ == "__main__":
    models = {
        "Without MSFA": build_without_msfa(),
        "Without LGFI": build_without_lgfi(),
        "Single MSFA Block": build_single_msfa()
    }

    for name, model in models.items():
        train_and_evaluate(model, name)
