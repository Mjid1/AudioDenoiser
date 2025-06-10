import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, LeakyReLU, MaxPooling2D, Dropout,
    concatenate, UpSampling2D
)
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)

# Bloc de convolution avec activation et dropout 
def conv_block(x, filters, dropout_rate=0.0):
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = LeakyReLU()(x)
    if dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
    return x

# Modèle U-Net++ (Nested U-Net) avec option poids pré-entraînés
def unet_plus_plus(input_size=(128, 128, 1), pretrained_weights=None):
    inputs = Input(input_size)

    # Niveau 0
    x00 = conv_block(inputs, 16)
    x10 = conv_block(MaxPooling2D()(x00), 32)
    x20 = conv_block(MaxPooling2D()(x10), 64)
    x30 = conv_block(MaxPooling2D()(x20), 128)
    x40 = conv_block(MaxPooling2D()(x30), 256, dropout_rate=0.5)

    # Niveau 1
    x01 = conv_block(concatenate([x00, UpSampling2D()(x10)], axis=3), 16)
    x11 = conv_block(concatenate([x10, UpSampling2D()(x20)], axis=3), 32)
    x21 = conv_block(concatenate([x20, UpSampling2D()(x30)], axis=3), 64)
    x31 = conv_block(concatenate([x30, UpSampling2D()(x40)], axis=3), 128)

    # Niveau 2
    x02 = conv_block(concatenate([x00, x01, UpSampling2D()(x11)], axis=3), 16)
    x12 = conv_block(concatenate([x10, x11, UpSampling2D()(x21)], axis=3), 32)
    x22 = conv_block(concatenate([x20, x21, UpSampling2D()(x31)], axis=3), 64)

    # Niveau 3
    x03 = conv_block(concatenate([x00, x01, x02, UpSampling2D()(x12)], axis=3), 16)
    x13 = conv_block(concatenate([x10, x11, x12, UpSampling2D()(x22)], axis=3), 32)

    # Niveau 4
    x04 = conv_block(concatenate([x00, x01, x02, x03, UpSampling2D()(x13)], axis=3), 16)

    # Sortie
    output = Conv2D(1, (1, 1), activation='tanh')(x04)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(), loss=tf.keras.losses.Huber(), metrics=['mae'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


