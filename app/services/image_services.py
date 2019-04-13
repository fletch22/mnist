import logging
import os

import numpy as np
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

from app.config import config
from app.services import file_services


def get_flow_from_directory(img_height, img_width, batch_size):
    gen_args = dict(featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                    zoom_range=0.1,  # Randomly zoom image
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=False,
                    rescale=1. / 255)

    train_datagen = ImageDataGenerator(gen_args)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_data_dir = os.path.join(config.DATA_FOLDER_PATH, 'mnist', 'train')
    test_data_dir = os.path.join(config.DATA_FOLDER_PATH, 'mnist', 'test')

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    train_size = len(file_services.walk_dir(train_data_dir, '.jpg'))
    test_size = len(file_services.walk_dir(test_data_dir, '.jpg'))

    return train_generator, validation_generator, train_size, test_size


def save_to_directory(X, y, dimension, folder_name):
    ndx = 0
    for numpy_array_image in X:
        image = Image.fromarray(np.uint8(numpy_array_image), mode='L')

        digit = str(y[ndx])
        digit_dir_path = os.path.join(config.DATA_FOLDER_PATH, 'mnist', folder_name, digit)
        os.makedirs(digit_dir_path, exist_ok=True)

        fp = os.path.join(digit_dir_path, f'{digit}_{ndx}.jpg')
        image.save(fp, format="JPEG")
        ndx += 1
        break


def show_images(datagen, image_size, X, y, num_channels):
    for X_batch, y_batch in datagen.flow(X, y, batch_size=9):
        # create a grid of 3x3 images
        for i in range(0, 9):
            pyplot.subplot(330 + 1 + i)
            logging.debug(f'Shape: {X_batch[i].shape}')

            if num_channels == 3:
                x_image = X_batch[i].reshape(image_size, image_size, num_channels)
            else:
                x_image = np.reshape(X_batch[i], (image_size, image_size))

            pyplot.imshow(x_image, cmap=pyplot.get_cmap('gray'))
        pyplot.show()
        break
