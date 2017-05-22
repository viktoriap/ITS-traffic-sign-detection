import os

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

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
            images.append(cv2.imread(image, 0).astype(np.float32) / 255.)
            labels.append(int(label))
    return images, labels


def resize_images(images, width=32, height=32):
    return [cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC) for image in images]


def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    # Fully connected layer.
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Define the loss function.
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # And, finally, an initialization op to execute before training.
    init = tf.global_variables_initializer()

session = tf.Session(graph=graph)
session.run(init)

# Load training data and resize
train_images, train_labels = load_data(training_dir)
train_images = resize_images(train_images)

# Load testing data and resize
test_images, test_labels = load_data(testing_dir)
test_images = resize_images(test_images)

for i in range(301):
    _, loss_value = session.run(
        [train, loss],
        feed_dict={images_ph: train_images, labels_ph: train_labels})
    if i % 10 == 0:
        print("Loss: ", loss_value)

# Run predictions against the full test set.
predicted = session.run([predicted_labels], feed_dict={images_ph: test_images, labels_ph: test_labels})[0]
# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = float(match_count) / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))

session.close()
