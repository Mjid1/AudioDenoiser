
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, LeakyReLU, MaxPooling2D, Dropout,
    concatenate, UpSampling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)

def conv_block(x, filters, dropout_rate=0.0):
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
    return x

def unet(input_size=(128, 128, 1), pretrained_weights=None, compile_model=True):
    inputs = Input(input_size)

    # Encodeur
    x00 = conv_block(inputs, 16)
    x10 = conv_block(MaxPooling2D()(x00), 32)
    x20 = conv_block(MaxPooling2D()(x10), 64)
    x30 = conv_block(MaxPooling2D()(x20), 128)
    x40 = conv_block(MaxPooling2D()(x30), 256, dropout_rate=0.5)

    # Decodeur 
    x01 = conv_block(concatenate([x00, UpSampling2D()(x10)], axis=3), 16)
    x11 = conv_block(concatenate([x10, UpSampling2D()(x20)], axis=3), 32)
    x21 = conv_block(concatenate([x20, UpSampling2D()(x30)], axis=3), 64)
    x31 = conv_block(concatenate([x30, UpSampling2D()(x40)], axis=3), 128)

    x02 = conv_block(concatenate([x00, x01, UpSampling2D()(x11)], axis=3), 16)
    x12 = conv_block(concatenate([x10, x11, UpSampling2D()(x21)], axis=3), 32)
    x22 = conv_block(concatenate([x20, x21, UpSampling2D()(x31)], axis=3), 64)

    x03 = conv_block(concatenate([x00, x01, x02, UpSampling2D()(x12)], axis=3), 16)
    x13 = conv_block(concatenate([x10, x11, x12, UpSampling2D()(x22)], axis=3), 32)

    x04 = conv_block(concatenate([x00, x01, x02, x03, UpSampling2D()(x13)], axis=3), 16)

    # Sorties intermediaires
    sd1 = Conv2D(1, (1, 1), activation='tanh', name='sd1')(x01)
    sd2 = Conv2D(1, (1, 1), activation='tanh', name='sd2')(x02)
    sd3 = Conv2D(1, (1, 1), activation='tanh', name='sd3')(x03)
    sd4 = Conv2D(1, (1, 1), activation='tanh', name='sd4')(x04)

    model = Model(inputs=inputs, outputs=[sd1, sd2, sd3, sd4])

    if compile_model:
        model.compile(
            optimizer=Adam(),
            loss={
                'sd1': tf.keras.losses.MeanSquaredError(),
                'sd2': tf.keras.losses.MeanSquaredError(),
                'sd3': tf.keras.losses.MeanSquaredError(),
                'sd4': tf.keras.losses.MeanSquaredError()
            },
            loss_weights={
                'sd1': 1.0,
                'sd2': 1.0,
                'sd3': 1.0,
                'sd4': 1.0
            }
        )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

