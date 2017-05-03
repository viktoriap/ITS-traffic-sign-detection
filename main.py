import os

import tensorflow as tf
import cv2
import numpy as np

training_dir = "dataset/Training/"
testing_dir = "dataset/Testing/"


def load_data(dir):
    subdirectories = [os.path.join(dir, o) for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]

    images = []
    labels = []
    for sub in subdirectories:
        files = [os.path.join(sub, o) for o in os.listdir(sub) if o.endswith(".ppm")]
        label = os.path.basename(sub)[:-1].lstrip("0") + os.path.basename(sub)[-1]
        for image in files:
            images.append(cv2.imread(image, 0))
            labels.append(int(label))
    return images, labels


def resize_images(images, width=32, height=32):
    for image in images:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return images


images, labels = load_data(training_dir)
resized_images = resize_images(images)
