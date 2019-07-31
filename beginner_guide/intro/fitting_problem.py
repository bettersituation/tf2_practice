from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


NUM_WORDS = 10000

train, test = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
train_data, train_labels = train
test_data, test_labels = test


def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])
plt.show()

baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

baseline_model.compile(
    optimizer='adam', loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

baseline_history = baseline_model.fit(
    train_data, train_labels,
    epochs=20, batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2)

smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

smaller_model.compile(
    optimizer='adam', loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

smaller_history = smaller_model.fit(
    train_data, train_labels,
    epochs=20, batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2)

bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

bigger_model.compile(
    optimizer='adam', loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy'])

bigger_model.summary()

bigger_history = bigger_model.fit(
    train_data, train_labels,
    epochs=20, batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2)


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
            '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key],
        color=val[0].get_color(), label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    return


model_histories = [
    ('baseline', baseline_history),
    ('smaller', smaller_history),
    ('bigger', bigger_history)
]

plot_history(model_histories)
plt.show()


l2_model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.001),
        input_shape=(NUM_WORDS, )),
    keras.layers.Dense(16, activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid')
])

l2_model.compile(
    optimizer='adam', loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
    epochs=20, batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2)


dropout_model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

dropout_model.compile(
    optimizer='adam', loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy'])

dropout_model_history = dropout_model.fit(train_data, train_labels,
    epochs=20, batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2)

model_histories.extend([('l2', l2_model_history), ('dropout', dropout_model_history)])
plot_history(model_histories)
plt.show()
