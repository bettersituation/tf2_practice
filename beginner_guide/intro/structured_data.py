from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


url = 'https://storage.googleapis.com/applied-dl/heart.csv'
df = pd.read_csv(url)
df = df.loc[df.thal.isin(['fixed', 'normal', 'reversible'])]

train_and_valid, test = train_test_split(df, test_size=0.2)
train, valid = train_test_split(train_and_valid, test_size=0.2)

print(len(train), 'train size')
print(len(valid), 'valid size')
print(len(test), 'test size')


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
valid_ds = df_to_dataset(valid, batch_size=batch_size)
test_ds = df_to_dataset(test, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print(f'all feature: {list(feature_batch.keys())} ({type(feature_batch)})')
    print(f'age values: {feature_batch["age"]}')
    print(f'label values: {label_batch}')

example_batch = next(iter(train_ds))[0]


def demo(feature_column):
    global example_batch
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


age = feature_column.numeric_column('age')
demo(age)

buckets = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
age_buckets = feature_column.bucketized_column(age, boundaries=buckets)
demo(age_buckets)

thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])

thal_onehot = feature_column.indicator_column(thal)
demo(thal_onehot)

thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000)
thal_hashhash = feature_column.indicator_column(thal_hashed)
demo(thal_hashhash)

'''
crossed_feature = feature_column.crossed_column(
    [age_buckets, thal],
    hash_bucket_size=1000)
crossed = feature_column.indicator_column(crossed_feature)
demo(crossed)
'''

feature_columns = []

for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

feature_columns.append(age_buckets)
# feature_columns.append(thal_onehot)
# feature_columns.append(thal_hashhash)
feature_columns.append(thal_embedding)
# feature_columns.append(crossed)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
valid_ds = df_to_dataset(valid, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(train_ds, validation_data=valid_ds, epochs=5)
model.summary()
