from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras


train, test = tf.keras.datasets.mnist.load_data()
train_images, train_labels = train
test_images, test_labels = test

train_images = train_images[:1000].reshape(-1, 28*28) / 255.
test_images = test_images[:1000].reshape(-1, 28*28) / 255.

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


model = create_model()
model.summary()

ckpt_path = 'save_load_py_1/cp.ckpt'
ckpt_dir = os.path.dirname(ckpt_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path, save_weights_only=True, verbose=1)

model.fit(train_images, train_labels, epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback])


# non-trained model
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print(f'non trained loss: {loss:.4f}, acc: {acc:.4f}')

model.load_weights(ckpt_path)
loss, acc = model.evaluate(test_images, test_labels)
print(f'restored loss: {loss:.4f}, acc: {acc:.4f}')

ckpt_path = 'save_load_py_2/cp-{epoch:04d}.ckpt'
ckpt_dir = os.path.dirname(ckpt_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path, save_weights_only=True, verbose=0, save_freq=5)

model = create_model()
model.save_weights(ckpt_path.format(epoch=0))

model.fit(train_images, train_labels, epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback], verbose=1)


# load ckpt
latest = tf.train.latest_checkpoint(ckpt_dir)
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print(f'restored loss: {loss:.4f}, acc: {100*acc:.2f}%')

h5fn = 'save_load.h5'
model.save(h5fn)
new_model = keras.models.load_model(h5fn)
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print(f'restored h5 model loss: {loss:.4f}, acc: {acc:.4f}')


import time

saved_model_path = 'save_load_py_saved_models/{}'.format(int(time.time()))
tf.keras.experimental.export_saved_model(model, saved_model_path)

new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()

if new_model.optimizer is None:
    print('saved model doesn\'t have optimizer info')
    optimizer = model.optimizer
else:
    optimizer = new_model.optimizer

new_model.compile(optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

loss, acc = new_model.evaluate(test_images, test_labels)
print(f'restored saved model loss: {loss:.4f}, acc: {acc:.4f}')
