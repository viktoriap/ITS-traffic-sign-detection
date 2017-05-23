# https://www.tensorflow.org/get_started/mnist/pros#train_and_evaluate_the_model
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
# https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6
import csv
import cv2
import os
import time
from datetime import timedelta
import argparse

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dataset locations.
training_dir = "dataset/Training/"
testing_dir = "dataset/Testing/"

# Data dimensions information.
img_size = 32
num_channels = 1
num_classes = 62

# CNN layers information.

# Convolutional Layer 1.
kernel_size1 = 5
num_filters1 = 6

# Convolutional Layer 2.
kernel_size2 = 5
num_filters2 = 16

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
    return [cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC) for image in images]


# Load training data and resize
train_images, train_labels = load_data(training_dir)
train_images = resize_images(train_images)
train_labels = np.eye(num_classes)[train_labels]

# Load testing data and resize
test_images, test_labels = load_data(testing_dir)
test_images = resize_images(test_images)
test_labels = np.eye(num_classes)[test_labels]


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


def train():
    graph = tf.Graph()

    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, img_size, img_size], name="images")
        x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
        y = tf.placeholder(tf.float32, [None, num_classes], name="labels")

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

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_fc2, labels=y))
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        predicted_labels = tf.argmax(h_fc2, 1, name="predicted")
        correct = tf.equal(predicted_labels, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        init = tf.global_variables_initializer()

    print "Training started ... "
    with tf.Session(graph=graph) as session:
        session.run(init)
        start_time = time.time()

        for i in range(301):
            session.run(optimizer, feed_dict={x: train_images,
                                              y: train_labels})
            # Print status every 10 iterations.
            if i % 10 == 0:
                # Calculate the accuracy.
                acc = session.run(accuracy, feed_dict={x: train_images,
                                                       y: train_labels})
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                print msg.format(i + 1, acc)

        end_time = time.time()
        print "Time usage: " + str(timedelta(seconds=int(round(end_time - start_time))))

        # Save the variables to disk.
        save_path = tf.train.Saver().save(session, "models/model-cnn.ckpt")
        print("Model saved in file: %s" % save_path)
        session.close()


def restore(filepaths):
    with tf.Session() as session:
        # Restore variables from disk.
        saver = tf.train.import_meta_graph('models/model-cnn.ckpt.meta')
        saver.restore(session, tf.train.latest_checkpoint('models/'))
        print "Model restored ... "

        session.run(tf.global_variables_initializer())

        images = []
        for f in filepaths:
            if os.path.isfile(f):
                image = cv2.imread(f, 0)
                images.append(image)
            else:
                print "No such file: %s" % f

        labels = []
        for f in filepaths:
            dir = f.split("/")[-2]
            label = int(dir[:-1].lstrip("0") + dir[-1])
            labels.append(label)

        images = resize_images(images)
        labels = np.eye(num_classes)[labels]

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("images:0")
        y = graph.get_tensor_by_name("labels:0")
        predicted_labels = graph.get_tensor_by_name("predicted:0")

        prediction = session.run([predicted_labels], feed_dict={x: images, y: labels})[0]
        labels = labels.nonzero()[1]
        return labels, prediction


def label_names(labels, filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        data = {int(rows[0]): rows[1] for rows in reader}
        output = []
        for key in labels:
            output.append(data.get(key))

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="The image file for which the user wants to know the traffic sign",
                        nargs="+",
                        required=True)
    parser.add_argument("-t", "--train",
                        help="Train the model, else use a previous model",
                        action="store_true")
    args = parser.parse_args()

    if args.train:
        train()

    labels, predictions = restore(args.input)
    labels_text = label_names(labels, "SignLabels.csv")
    prediction_text = label_names(predictions, "SignLabels.csv")

    print "Correct label\t-\tPrediction"
    for i in range(len(labels_text)):
        print labels_text[i], "\t-\t ", prediction_text[i]
