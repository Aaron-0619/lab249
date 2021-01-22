import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
# from keras.callbacks import EarlyStopping
from keras import optimizers
import numpy as np
import pandas as pd
import keras
NUM_CLASSES = 12

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, growth_rate, drop_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=4 * growth_rate,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=growth_rate,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.dropout(x, training=training)
        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.features_list = []

    def _make_layer(self, x, training):
        y = BottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate)(x, training=training)
        self.features_list.append(y)
        y = tf.concat(self.features_list, axis=-1)
        return y


    def call(self, inputs, training=None, **kwargs):
        self.features_list.append(inputs)
        x = self._make_layer(inputs, training=training)
        for i in range(1, self.num_layers):
            x = self._make_layer(x, training=training)
        self.features_list.clear()
        return x


class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filters=out_channels,
                                           kernel_size=(1, 1),
                                           strides=1,
                                           padding="same")
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                              strides=2,
                                              padding="same")


    def call(self, inputs, training=None, **kwargs):
        x = self.bn(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(tf.keras.Model):
    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate):
        super(DenseNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=num_init_features,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2,
                                              padding="same")
        self.num_channels = num_init_features
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[0]
        self.num_channels = compression_rate * self.num_channels
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))
        self.dense_block_2 = DenseBlock(num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))
        self.dense_block_3 = DenseBlock(num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))
        self.dense_block_4 = DenseBlock(num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)


    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.dense_block_1(x, training=training)
        x = self.transition_1(x, training=training)
        x = self.dense_block_2(x, training=training)
        x = self.transition_2(x, training=training)
        x = self.dense_block_3(x, training=training)
        x = self.transition_3(x, training=training)
        x = self.dense_block_4(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def densenet_121(): #45
    return DenseNet(num_init_features=32, growth_rate=32, block_layers=[5, 5, 5, 5, 5], compression_rate=0.5, drop_rate=0.5)

if __name__ == '__main__':
    model = densenet_121()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    Data_Path = "D:\\aaronliao\\FD-CNN-master\\USC-HAD\\Train.csv"
    data = pd.read_csv(Data_Path, index_col=None)
    train_x = data.iloc[0:5603, 1:1201]
    xx = train_x.iloc[0:5603, ::3]
    yy = train_x.iloc[0:5603, 1::3]
    zz = train_x.iloc[0:5603, 2::3]
    tt = np.concatenate([xx, yy, zz], -1)
    train_x = np.reshape(tt, (5603, 3, 20, 20))
    train_label = data['label']
    train_label = np.array(train_label)
    train_label = keras.utils.to_categorical(train_label - 1)

    Data_Path2 = "D:\\aaronliao\\FD-CNN-master\\USC-HAD\\Test.csv"
    data = pd.read_csv(Data_Path2, index_col=None)
    test_x = data.iloc[0:1401, 1:1201]
    xxx = test_x.iloc[0:1401, ::3]
    yyy = test_x.iloc[0:1401, 1::3]
    zzz = test_x.iloc[0:1401, 2::3]
    ttt = np.concatenate([xxx, yyy, zzz], -1)
    test_x = np.reshape(ttt, (1401, 3, 20, 20))
    test_label = data['label']
    test_label = np.array(test_label)
    test_label = keras.utils.to_categorical(test_label - 1)
    train_x = tf.cast(train_x, tf.float32)
    test_x = tf.cast(test_x, tf.float32)

    verbose, epochs, batch_size = 1, 50000, 500
    earlystopping = EarlyStopping(monitor="loss", min_delta=0.000001, patience=100, verbose=1, mode='auto')
    tf.config.experimental_run_functions_eagerly(True)
    history = model.fit(train_x, train_label, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(test_x, test_label), callbacks=[earlystopping], shuffle=True, use_multiprocessing=True, workers=8)

    _, accuracy = model.evaluate(test_x, test_label, batch_size=batch_size, verbose=verbose)
    print('\nTesting loss: {}, acc: {}\n'.format(_, accuracy))
