import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import sklearn
import time
import PIL, io

from tensorflow.keras.utils import Progbar

def get_dataset_len(dataset):
    return int(tf.data.experimental.cardinality(dataset))


def random_erasing(img, probability=0.5, sl=0.02, sh=0.3, r1=0.3, method='random'):
    # Motivated by https://github.com/Amitayus/Random-Erasing-TensorFlow.git
    # Motivated by https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    img : 3D Tensor data (H,W,Channels) normalized value [0,1]
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    method : 'black', 'white' or 'random'. Erasing type
    -------------------------------------------------------------------------------------
    '''
    assert method in ['random', 'white', 'black'], 'Wrong method parameter'

    if tf.random.uniform([]) > probability:
        return img

    img_width = img.shape[1]
    img_height = img.shape[0]
    img_channels = img.shape[2]

    area = img_height * img_width

    target_area = tf.random.uniform([], minval=sl, maxval=sh) * area
    aspect_ratio = tf.random.uniform([], minval=r1, maxval=1 / r1)
    h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
    w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

    while tf.constant(True, dtype=tf.bool):
        if h > img_height or w > img_height:
            target_area = tf.random.uniform([], minval=sl, maxval=sh) * area
            aspect_ratio = tf.random.uniform([], minval=r1, maxval=1 / r1)
            h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
            w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)
        else:
            break

    x1 = tf.cond(img_height == h, lambda: 0,
                 lambda: tf.random.uniform([], minval=0, maxval=img_height - h, dtype=tf.int32))
    y1 = tf.cond(img_width == w, lambda: 0,
                 lambda: tf.random.uniform([], minval=0, maxval=img_width - w, dtype=tf.int32))

    part1 = tf.slice(img, [0, 0, 0], [x1, img_width, img_channels])  # first row
    part2 = tf.slice(img, [x1, 0, 0], [h, y1, img_channels])  # second row 1

    if method is 'black':
        part3 = tf.zeros((h, w, img_channels), dtype=tf.float32)  # second row 2
    elif method is 'white':
        part3 = tf.ones((h, w, img_channels), dtype=tf.float32)
    elif method is 'random':
        part3 = tf.random.uniform((h, w, img_channels), maxval=255, dtype=tf.float32)

    part4 = tf.slice(img, [x1, y1 + w, 0], [h, img_width - y1 - w, img_channels])  # second row 3
    part5 = tf.slice(img, [x1 + h, 0, 0], [img_height - x1 - h, img_width, img_channels])  # third row

    middle_row = tf.concat([part2, part3, part4], axis=1)
    img = tf.concat([part1, middle_row, part5], axis=0)

    return img


def _parse_data_train(filename_table, transform_image_pixels, config):
    '''
        filename_table is used to get filename of image at index i
        faceloc_table is used to get 4 facial bounding box coordinates (x,y,w,h) of image at index i

    '''

    def parse_data(file_path, valence, arousal, label, knn, knn_weights, idx):
        num_neighbors = config.num_neighbors
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)  # full raw_image

        va_regression_true = tf.convert_to_tensor([valence, arousal])  # shape (2,)

        # parse the k-nn from filenames to image tensor
        neighbor_index = tf.strings.split(knn, sep=';')
        neighbor_index = tf.strings.to_number(neighbor_index, out_type=tf.int32)

        neighbor_index = neighbor_index[:num_neighbors]
        knn_weights = tf.cast(knn_weights[:num_neighbors], tf.float32)
        neighbor_filenames = filename_table.lookup(neighbor_index)
        neighbor_images = []
        # loop thourgh each neighbor (with neighbor_index)
        for i in range(num_neighbors):
            neighbor_image = tf.io.read_file(neighbor_filenames[i])
            neighbor_image = tf.image.decode_image(neighbor_image, channels=3, expand_animations=False)

            neighbor_image = tf.image.resize(neighbor_image, config.input_size)
            neighbor_image = tf.image.random_flip_left_right(neighbor_image)
            neighbor_image = tf.pad(neighbor_image, paddings=[[config.pad_size, config.pad_size], [config.pad_size, config.pad_size], [0, 0]])
            neighbor_image = tf.image.random_crop(neighbor_image, size=config.input_size + [3])
            neighbor_image = transform_image_pixels(neighbor_image)

            neighbor_images.append(neighbor_image)

        neighbor_images = tf.stack(neighbor_images,
                                   axis=0)  # a tensor with shape (K, 112,112,3) where K is the number of neighbors

        return image, va_regression_true, label, neighbor_images, knn_weights, idx, neighbor_index

    return parse_data


def _training_preprocess(transform_image_pixels, config):
    def training_preprocess(image, va_regression_true, label, neighbor_images, knn_weights, idx, neighbor_index):
        image = tf.image.resize(image, config.input_size)
        image = tf.image.random_flip_left_right(image)
        image = tf.pad(image, paddings=[[config.pad_size, config.pad_size], [config.pad_size, config.pad_size], [0, 0]])
        image = tf.image.random_crop(image, size=config.input_size + [3])
        image = random_erasing(image, probability=0.5)
        image = transform_image_pixels(image)
        return image, va_regression_true, label, neighbor_images, knn_weights, idx, neighbor_index

    return training_preprocess

def _parse_data_test(config):
    def parse_data_test(file_path, valence, arousal, label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)  # full raw_image
        va_regression_true = tf.convert_to_tensor([valence, arousal])  # shape (2,)
        image = tf.image.resize(image, config.input_size)
        return image, va_regression_true, label
    return parse_data_test


def _testing_preprocess(transform_image_pixels, config):
    def testing_preprocess(image, va_regression_true, label):

        image = tf.pad(image, paddings=[[config.pad_size, config.pad_size], [config.pad_size, config.pad_size], [0, 0]])
        image = tf.image.central_crop(image, config.input_size[0] / (config.input_size[0] + 2 * config.pad_size))
        image = transform_image_pixels(image)

        return image, va_regression_true, label

    return testing_preprocess


def _transform_image_pixels(mean=None, std=None, center=True, scale=False, BGR=False):
    def transform_image_pixels(image):
        if mean is not None:  # most of resnet default style (caffe)
            if BGR:
                image = tf.reverse(image, axis=[-1])
            if std is not None:
                image = (tf.cast(image, dtype=tf.float32) / 255.0 - mean) / std
            else:
                image = tf.cast(image, dtype=tf.float32) - mean

        else:
            if center:
                image = (tf.cast(image, dtype=tf.float32) - 127.5)
                if scale:
                    image = image / 127.5
            else:  # center=False
                image = tf.cast(image, dtype=tf.float32)
                if scale:
                    image = tf.cast(image, dtype=tf.float32) / 255.0
        return image

    return transform_image_pixels

def get_train_dataset(train_data_path, image_dir, config):
    """
    train_data_path: path to csv annotation file,
    the csv should have following columns:
    subDirectory_filePath, expression, valence, arousal, knn
    """

    train_data = pd.read_csv(train_data_path)
    train_data['subDirectory_filePath'] = image_dir + os.sep +  train_data['subDirectory_filePath']

    # pre-computed local similarity between central image and neighbors
    neighbors_index = np.array(train_data['knn'].str.split(";").to_list()).astype(np.int32)
    central_va = train_data[['valence', 'arousal']].to_numpy()  # shape(N,2)
    neighbor_va = central_va[neighbors_index]  # shape (N,K,2)
    dist = central_va - neighbor_va.transpose((1, 0, 2))  # shape (K,N,2)
    dist = dist.transpose((1, 0, 2))  # shape (N,K,2)
    dist = np.sum(np.abs(dist), axis=-1)  # shape (N,K) distance between central i-th  and neighbor j-th
    knn_weights = np.exp(-dist / 0.5)

    # create tf.dataset
    keys_tensor = tf.convert_to_tensor(np.arange(len(train_data)), dtype=tf.int32)
    vals_tensor = tf.convert_to_tensor(train_data['subDirectory_filePath'], dtype=tf.string)
    # a table that used for looking up image's filename at index i
    filename_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value="None")

    #load from csv data
    list_dataset = tf.data.Dataset.from_tensor_slices((train_data['subDirectory_filePath'],
                                                       train_data['valence'],
                                                       train_data['arousal'],
                                                       train_data['expression'].astype(np.int32),
                                                       train_data['knn'],
                                                       knn_weights,  # local similarity a_ij of central i and neighbor j
                                                       np.arange(len(train_data), dtype=np.int32)
                                                       ))
    list_dataset = list_dataset.shuffle(get_dataset_len(list_dataset))

    transform_image_pixels_func = _transform_image_pixels(center=True, scale=True)
    # transform_image_pixels_func = _transform_image_pixels(mean=[0.57535914, 0.44928582, 0.40079932],
    #                                                       std=[0.20735591, 0.18981615, 0.18132027])
    training_preprocess = _training_preprocess(transform_image_pixels_func, config)

    train_dataset = list_dataset.shuffle(1024, reshuffle_each_iteration=True). \
        map(_parse_data_train(filename_table, transform_image_pixels_func, config), config.num_parallel_calls). \
        map(training_preprocess, config.num_parallel_calls).batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset


def get_test_dataset(test_data_path, image_dir, config):
    test_data = pd.read_csv(test_data_path)
    test_data['subDirectory_filePath'] = image_dir + os.sep +  test_data['subDirectory_filePath']

    transform_image_pixels_func = _transform_image_pixels(center=True, scale=True)
    testing_preprocess = _testing_preprocess(transform_image_pixels_func, config)
    test_list_dataset = tf.data.Dataset.from_tensor_slices((test_data['subDirectory_filePath'],
                                                            tf.ones(len(test_data)),  # test_data['valence'], just a dummy value
                                                            tf.ones(len(test_data)),  # test_data['arousal'],
                                                            test_data['expression']
                                                            ))
    test_dataset = test_list_dataset.map(_parse_data_test(config), config.num_parallel_calls). \
        map(testing_preprocess, config.num_parallel_calls). \
        batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return test_dataset

if __name__ == '__main__':
    import sys
    sys.path.append("cfg_files")
    config = __import__("config_resnet18_raf").config

    train_dataset = get_train_dataset("data/rafdb/raf_train.csv", image_dir="data/rafdb/aligned", config=config)
    print(get_dataset_len(train_dataset))

    import matplotlib.pyplot as plt

    idx = 5
    for i in train_dataset.take(1):
        i = i[:-2]
        print(i[0].shape)
        plt.figure()
        plt.imshow((i[0][idx] + 1) / 2)

        plt.figure()
        print(i[-2].shape)
        print(i[-1].shape)
        for j, aux_i in enumerate(i[-2][idx]):
            plt.subplot(1, 8, j + 1)
            plt.imshow((aux_i + 1) / 2)
            plt.axis('off')

    plt.show()

    print("="*50)
    test_dataset = get_test_dataset("data/rafdb/test.csv", image_dir="data/rafdb/aligned", config=config)
    print(get_dataset_len(test_dataset))
    idx = 5
    for i in test_dataset.take(1):
        print(i[0].shape)
        plt.figure()
        plt.imshow((i[0][idx] + 1) / 2)
    plt.show()
