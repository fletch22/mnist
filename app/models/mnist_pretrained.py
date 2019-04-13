import warnings
warnings.filterwarnings('ignore')

import logging

from keras import Model, Input
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Convolution2D, Activation, MaxPooling2D, Reshape, BatchNormalization, LeakyReLU


def get_vgg16_for_mnist(img_shape, num_classes):
    logging.info(f'Pretrained_model shape: {img_shape}')

    layer_type = 'relu'
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    # model_vgg16_conv.summary()

    toggle_training_in_layers(model_vgg16_conv)

    # Create your own input format
    keras_input = Input(shape=img_shape, name='image_input')

    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(keras_input)

    # Add the fully-connected layers
    # x = Flatten(name='flatten')(output_vgg16_conv)
    # x = Dense(4096, activation=layer_type, name='fc1')(x)
    # x = Dense(4096, activation=layer_type, name='fc2')(x)

    x = Reshape((16,32,1), input_shape=(1,1,512))(output_vgg16_conv)

    # x = Convolution2D(32, (3, 3))(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    #
    # x = Convolution2D(32, (3, 3))(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    from keras import backend as K

    LeakyReLU(alpha=0.3)

    x = Convolution2D(64, (3, 3))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(64, (3, 3))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten(name='flatten')(x)

    x = Dense(512, activation=layer_type, name='fc1')(x)
    Dense(512),
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create your own model
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    flatten_layer = pretrained_model.get_layer("flatten")
    logging.info(f'Layer output: {flatten_layer.input_shape}')

    return pretrained_model


def toggle_training_in_layers(model):
    for layer in model.layers:
        layer.trainable = False
