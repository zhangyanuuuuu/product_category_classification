import requests
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from torch import nn
from keras.applications.vgg16 import VGG16


def get_image_array(data, labels, label_map):
    scaled = nn.AdaptiveMaxPool2d((1200,1200))
    x_train = np.zeros((len(labels), 1200*1200))
    y_train = np.array([label_map[label] for label in labels])
    x_test = np.zeros((len(data) - len(labels), 1200*1200))
    for i, row in enumerate(data):
        image_address = row['image_url']
        image = requests.get(image_address)
        grey_image = Image.open(BytesIO(image.content)).convert("L")
        grey_image_arr = np.array(grey_image)
        temp = torch.rand(1, grey_image_arr.shape[0], grey_image_arr.shape[1])
        temp[0] = torch.from_numpy(grey_image_arr)
        output = scaled(temp)[0].numpy()
        flattened = np.ravel(output)
        if i < len(labels):
            x_train[i] = flattened
        else:
            x_test[i - len(labels)] = flattened
    return x_train, y_train, x_test


def get_image_data(data, labels, label_map, batch_size=16):
    scaled = nn.AdaptiveMaxPool2d((1200, 1200))
    x_train = np.zeros((len(labels), 37*37*512))
    y_train = np.array([label_map[label] for label in labels])
    x_test = np.zeros((len(data) - len(labels), 37*37*512))
    image_batch = np.zeros((batch_size, 3, 1200, 1200))
    # load model
    model = VGG16(include_top=False, input_shape=(1200,1200,3))
    for i, row in enumerate(data):
        image_address = row['image_url']
        image = requests.get(image_address)
        rgb = np.array(Image.open(BytesIO(image.content)).convert("RGB")).astype(np.float32)
        channel_first = np.rollaxis(np.array(rgb), 2, 0)
        scaled_image = scaled(torch.from_numpy(channel_first)).numpy()
        image_batch[i % batch_size] = scaled_image
        if i % batch_size == batch_size - 1:
            channels_last = np.rollaxis(image_batch, 1, 4)
            last_layers = model.predict_on_batch(channels_last)
            batch_num = i // batch_size
            for j in range(batch_size):
                index = batch_num * batch_size + j
                if index < len(labels):
                    x_train[index] = last_layers[j].ravel()
                else:
                    x_test[index - len(labels)] = last_layers[j].ravel()
    return x_train, y_train, x_test


def save_image_training_data(data, categories, labels):
    # image based classifier
    label_map = {}
    for i, category in enumerate(categories):
        label_map[category.lower()] = i
    x_train, y_train, x_test = get_image_data(data, labels, label_map)
    with open('x_train.npy', 'wb') as f:
        np.save(f, x_train)
    with open('y_train.npy', 'wb') as f:
        np.save(f, y_train)
    with open('x_test.npy', 'wb') as f:
        np.save(f, x_test)