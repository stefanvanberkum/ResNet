#!/usr/bin/env python
# coding: utf-8

import csv
import os
import time
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorflow import pad
from tensorflow.data import AUTOTUNE
from tensorflow.image import stateless_random_crop, stateless_random_flip_left_right
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import relu
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Normalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.random.experimental import stateless_split
from tensorflow_addons.layers import StochasticDepth


def main(dataset, n, p_L):
    # Make directories.
    os.makedirs('cifar10/raw', exist_ok=True)
    os.makedirs('cifar100/raw', exist_ok=True)

    epochs = 100

    if dataset == 'cifar10':
        # Load CIFAR-10 data.
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        path = 'cifar10/'
        n_out = 10
    else:
        # Load CIFAR-100 data.
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        path = 'cifar100/'
        n_out = 100

    if not os.path.isfile(path + 'results.csv'):
        with open(path + 'results.csv', 'w') as results:
            writer = csv.writer(results)
            writer.writerow(['n', 'p_L', 'acc', 'time (s)', 'time (min)'])

    # Initialize normalization layer.
    norm_layer = Normalization(axis=(1, 2, 3), mean=X_train.mean(axis=0), variance=1)

    # Convert from integers to floats.
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Apply normalization.
    X_train = norm_layer(X_train)
    X_test = norm_layer(X_test)

    # Data augmentation.
    # Pad X_train.
    X_train = pad(X_train, [[0, 0], [4, 4], [4, 4], [0, 0]])

    def augment(image_label, seed):
        """Augmentation function."""
        image, label = image_label

        # Make a new seed.
        new_seed = stateless_split(seed, num=1)[0, :]

        # Randomly flip and crop.
        image = stateless_random_flip_left_right(image, seed=seed)
        image = stateless_random_crop(image, size=[32, 32, 3], seed=new_seed)
        return image, label

    rng = tf.random.Generator.from_seed(123, alg='philox')

    def f(x, y):
        """Wrapper function."""
        seed = rng.make_seeds(2)[0]
        image, label = augment((x, y), seed)
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1280, reshuffle_each_iteration=True).map(f, num_parallel_calls=AUTOTUNE).batch(
        128).prefetch(AUTOTUNE)

    def save_result(n, p_L, history, t, path):
        """Evaluate the model and store the results.

        :param n: Network size parameter.
        :param p_L: Final survival probability for linear decay.
        :param history: Model training history object.
        :param t: Training time.
        :param path: Path to results.
        """

        _, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test accuracy: %.3f' % (acc * 100.0))
        with open(path + 'results.csv', 'a') as results:
            writer = csv.writer(results)
            writer.writerow([n, p_L, round(100 * acc, 2), round(t), round(t / 60)])

        prefix = f"{path}raw/{n}_{int(100 * p_L)}_"

        train_loss_path = prefix + 'train_loss.npy'
        test_loss_path = prefix + 'test_loss.npy'
        train_acc_path = prefix + 'train_acc.npy'
        test_acc_path = prefix + 'test_acc.npy'

        with open(train_loss_path, 'wb') as f:
            np.save(f, history.history['loss'])

        with open(test_loss_path, 'wb') as f:
            np.save(f, history.history['val_loss'])

        with open(train_acc_path, 'wb') as f:
            np.save(f, history.history['accuracy'])

        with open(test_acc_path, 'wb') as f:
            np.save(f, history.history['val_accuracy'])

    def ResNet(n, n_out, p_L, weight_decay=1e-4, bottleneck=False):
        """ResNet for CIFAR-10 as described by He et al. (2015).

        This network has 2n + 2 layers.

        :param n: Network size parameter.
        :param n_out: Output size.
        :param p_L: Final survival probability for linear decay.
        :param weight_decay: Weight decay parameter.
        :param bottleneck: True if bottleneck blocks should be used.
        """

        stages = [16, 32, 64]
        inputs = Input(shape=(32, 32, 3))

        n_blocks = 3 * n
        block_count = 1

        # First convolution.
        x = Conv2D(16, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(inputs)
        x = BatchNormalization()(x)
        x = relu(x)

        # ResNet blocks.
        for i in range(len(stages)):
            n_filters = stages[i]
            for j in range(n):
                # Compute survival probability using linear decay.
                p_s = 1 - block_count / n_blocks * (1 - p_L)

                # Add ResNet block.
                if bottleneck:
                    # Use bottleneck blocks.
                    if i == 0 and j == 0:
                        x = bottleneck_block(x, n_filters, p_s, weight_decay, first=True)
                    elif j == 0:
                        x = bottleneck_block(x, n_filters, p_s, weight_decay, downsample=True)
                    else:
                        x = bottleneck_block(x, n_filters, p_s, weight_decay)
                else:
                    # Use classic ResNet blocks.
                    if i > 0 and j == 0:
                        x = block(x, n_filters, p_s, weight_decay, downsample=True)
                    else:
                        x = block(x, n_filters, p_s, weight_decay)
                block_count += 1

        # Pooling and dense output layer with softmax activation.
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(n_out, activation='softmax', kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay))(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile model.
        opt = SGD(learning_rate=0.1, momentum=0.9)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def block(x, n_filters, p_s, weight_decay, downsample=False, use_conv=False):
        """Classic ResNet block.

        :param x: Input.
        :param n_filters: Number of filters.
        :param p_s: Survival probability.
        :param weight_decay: Weight decay parameter.
        :param downsample: True if the layer should downsample.
        :param use_conv: True if a convolution operation should be used to match residual dimensions on downsampling.
        """

        if downsample:
            start_stride = 2
        else:
            start_stride = 1

        x_skip = x

        x = Conv2D(n_filters, 3, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)

        if downsample:
            if use_conv:
                x_skip = Conv2D(n_filters, 1, 2, use_bias=False, kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay))(x_skip)
                x_skip = BatchNormalization()(x_skip)
            else:
                x_skip = x_skip[:, ::2, ::2, :]
                missing = n_filters - x_skip.shape[3]
                x_skip = pad(x_skip, [[0, 0], [0, 0], [0, 0], [missing // 2, -(missing // -2)]])

        if p_s == 1:
            x = Add()([x_skip, x])
        else:
            x = StochasticDepth(p_s)([x_skip, x])
        x = relu(x)

        return x

    def bottleneck_block(x, n_filters, p_s, weight_decay, first=False, downsample=False, use_conv=False):
        """Bottleneck ResNet block.

        :param x: Input.
        :param n_filters: Number of filters.
        :param p_s: Survival probability.
        :param weight_decay: Weight decay parameter.
        :param first: True if this is the first block.
        :param downsample: True if the layer should downsample.
        :param use_conv: True if a convolution operation should be used to match residual dimensions on downsampling.
        """

        if downsample:
            start_stride = 2
        else:
            start_stride = 1

        x_skip = x

        x = Conv2D(n_filters, 1, start_stride, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(n_filters, 3, 1, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(4 * n_filters, 1, 1, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)

        if downsample:
            if use_conv:
                x_skip = Conv2D(4 * n_filters, 1, 2, kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay))(x_skip)
                x_skip = BatchNormalization()(x_skip)
            else:
                x_skip = x_skip[:, ::2, ::2, :]
                missing = n_filters - x_skip.shape[3]
                x_skip = pad(x_skip, [[0, 0], [0, 0], [0, 0], [missing // 2, -(missing // -2)]])
        elif first:
            if use_conv:
                x_skip = Conv2D(4 * n_filters, 1, 1, kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay))(x_skip)
                x_skip = BatchNormalization()(x_skip)
            else:
                missing = n_filters - x_skip.shape[3]
                x_skip = pad(x_skip, [[0, 0], [0, 0], [0, 0], [missing // 2, -(missing // -2)]])

        if p_s == 1:
            x = Add()([x_skip, x])
        else:
            x = StochasticDepth(p_s)([x_skip, x])
        x = relu(x)

        return x

    def scheduler(epoch, lr):
        """Learning rate scheduler."""
        if epoch == 82:
            return lr / 10
        else:
            return lr

    # Run experiment.
    print("")
    print(f"Running ResNet on {dataset} with n = {n} and p_L = {p_L}...")
    model = ResNet(n, n_out, p_L)
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    t_start = time.time()
    history = model.fit(train_ds, epochs=epochs, validation_data=(X_test, y_test), callbacks=[callback], verbose=2)
    t = time.time() - t_start
    save_result(n, p_L, history, t, path)
    print("")


if __name__ == '__main__':
    # For command line use.
    parser = ArgumentParser()
    parser.add_argument('dataset', help='The dataset, one of {cifar10, cifar100}.')
    parser.add_argument('n', help='Network size parameter.', type=int)
    parser.add_argument('p_L', help='Final survival probability for linear decay.', type=float)
    args = parser.parse_args()

    main(args.dataset, args.n, args.p_L)
