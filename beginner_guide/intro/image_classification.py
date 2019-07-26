"""tf2 image classification for beginner
this is an illustration for how to use keras
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(precision=4, suppress=True)


fashion_mnist = keras.datasets.fashion_mnist
train_data, test_data = fashion_mnist.load_data()

train_images, train_labels = train_data
test_images, test_labels = test_data

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
]

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 1
model.fit(train_images, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'test_loss {test_loss:.4f} acc {test_acc:.4f}')

predictions = model.predict(test_images)
print(predictions[0])

predict_label = class_names[np.argmax(predictions[0])]
true_label = class_names[test_labels[0]]
print(f'predict: {predict_label} truly: {true_label}')


def plot_image(i, predictions_array, true_label, img):
    predictions_array = predictions_array[i]
    true_label = true_label[i]
    img = img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    predicted_class = class_names[predicted_label]
    predicted_pct = 100 * np.max(predictions_array)
    true_class = class_names[true_label]

    description = f'{predicted_class} {predicted_pct:.2f}% ({true_class})'

    plt.xlabel(description, color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array = predictions_array[i]
    true_label = true_label[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    distplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    distplot[predicted_label].set_color('red')
    distplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()
