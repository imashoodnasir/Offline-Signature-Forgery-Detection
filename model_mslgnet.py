# model_mslgnet.py

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Multiply, Add, Concatenate, Reshape
from tensorflow.keras.models import Model

def msfa_block(x):
    gap1 = GlobalAveragePooling2D()(x)
    pool = AveragePooling2D(pool_size=(2, 2))(x)
    gap2 = GlobalAveragePooling2D()(pool)
    concat = Concatenate()([gap1, gap2])
    dense1 = Dense(64, activation='relu')(concat)
    dense2 = Dense(x.shape[-1], activation='sigmoid')(dense1)
    reshaped = Reshape((1, 1, x.shape[-1]))(dense2)
    return Multiply()([x, reshaped])

def lgfi_block(local_features, global_features):
    dense = Dense(local_features.shape[-1])(global_features)
    reshaped = Reshape((1, 1, local_features.shape[-1]))(dense)
    norm = BatchNormalization()(reshaped)
    act = Activation('relu')(norm)
    multiplied = Multiply()([local_features, act])
    return Add()([local_features, multiplied])

def build_mslg_net(input_shape=(224, 224, 1)):
    inputs = Input(shape=input_shape)

    # Initial Block
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # First MSFA + Conv
    x = msfa_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # LGFI Block
    global_feat = GlobalAveragePooling2D()(x)
    x = lgfi_block(x, global_feat)

    # Second MSFA + Final Conv
    x = msfa_block(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='MSLGNet')
    return model
