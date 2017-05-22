import random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# https://www.tensorflow.org/get_started/mnist/pros#train_and_evaluate_the_model
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb

# Dataset locations.
training_dir = "dataset/Training/"
testing_dir = "dataset/Testing/"

# Data dimensions information.
img_size = 32
img_size_flat = img_size * img_size
num_channels = 1
num_classes = 62

# CNN layers information.
# Convolutional Layer 1.
kernel_size1 = 3
num_filters1 = 16

# Convolutional Layer 2.
kernel_size2 = 3
num_filters2 = 36

# Fully-connected layer.
fc_size = 128  # Number of neurons in fully-connected layer.


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
    output = []
    for image in images:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        output.append(image)
    return output


# Load training data and resize
train_images, train_labels = load_data(training_dir)
train_images = resize_images(train_images)

# Load testing data and resize
test_images, test_labels = load_data(testing_dir)
test_images = resize_images(test_images)


def make_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def make_biases(shape):
    return tf.Variable(tf.constant(0.1, shape=[shape]))


def convolution(in_data, kernel_size, num_channels, depth):
    weights = make_weights([kernel_size, kernel_size, num_channels, depth])
    biases = make_biases(depth)

    conv_layer = tf.nn.conv2d(in_data, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
    conv_layer = tf.nn.relu(conv_layer + biases)
    return conv_layer


def pool(in_data):
    # Max-pooling with 2x2 kernel and stride 2.
    pooled = tf.nn.max_pool(in_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return pooled


def fully_connected(in_data, num_inputs, num_outputs):
    weights = make_weights([num_inputs, num_outputs])
    biases = make_biases(num_outputs)

    fc_layer = tf.nn.relu(tf.matmul(in_data, weights) + biases)
    return fc_layer


graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder(tf.float32, [None, img_size, img_size])
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    y = tf.placeholder(tf.float32, [None, num_classes])

    # Convolution and maxpooling.
    h_conv1 = convolution(x_image, kernel_size1, num_channels, num_filters1)
    h_pool1 = pool(h_conv1)

    h_conv2 = convolution(h_pool1, kernel_size2, num_filters1, num_filters2)
    h_pool2 = pool(h_conv2)
    shape = h_pool2.get_shape().as_list()
    h_pool2_flat = tf.reshape(h_pool2, [-1, np.prod(shape[1:4])])

    # Fully-connected.
    h_fc1 = fully_connected(h_pool2_flat, np.prod(shape[1:4]), fc_size)
    h_fc2 = fully_connected(h_fc1, fc_size, num_classes)

    pred = tf.argmax(h_fc2, 1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_fc2, labels=y))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct = tf.equal(pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

session = tf.Session(graph=graph)
session.run(init)

for i in range(201):
    # x_batch, y_true_batch = data.train.next_batch(50)
    _, loss_value = session.run(
        [train, loss],
        feed_dict={x: train_images, y: train_labels})
    if i % 10 == 0:
        print("Loss: ", loss_value)

# Pick 10 random images
sample_indexes = random.sample(range(len(train_images)), 10)
sample_images = [train_images[i] for i in sample_indexes]
sample_labels = [train_labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = session.run(pred,
                        {x: sample_images})
print(sample_labels)
print(predicted)

session.close()
