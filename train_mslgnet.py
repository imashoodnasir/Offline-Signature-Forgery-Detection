# train_mslgnet.py

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from model_mslgnet import build_mslg_net

# Load dataset
data = np.load('step1_dataset_prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Build and compile model
model = build_mslg_net(input_shape=(224, 224, 1))
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('mslgnet_best_model.h5', save_best_only=True)
]

# Train model
model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, callbacks=callbacks)

# Save final model
model.save('mslgnet_final_model.h5')
