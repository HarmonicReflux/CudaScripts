#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


import numpy as np
import  matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1  # ensure each class has at least 1 pair
    for d in range(10):
        for i in range(n):
            # positive pair: same digit, consecutive images
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            labels += [1]  # label for positive pair is 1
            # negative pair: different digits
            inc = random.randrange(1, 10)  # randomly pick a different digit
            dn = (d + inc) % 10  # cycle through digits to avoid index out of range
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]  # different class
            pairs += [[x[z1], x[z2]]]
            labels += [0]  # label for negative pair is 0
    return np.array(pairs), np.array(labels)


def create_pairs_on_set(images, labels):
    digit_indices = [np.where(labels==i)[0] for i in range(10)]
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype('float32')
    return pairs, y


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.astype('float32')
test_images = test_images.astype('float32') 
train_images = train_images / 255
test_images = test_images / 255

tr_pairs, tr_y = create_pairs_on_set(train_images, train_labels)
ts_pairs, ts_y = create_pairs_on_set(test_images, test_labels)


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()




this_pair = 10
show_image(tr_pairs[this_pair][0])
show_image(tr_pairs[this_pair][1])
print(tr_y[this_pair])




def initialize_base_network():
    input = Input(shape=(28,28,), name='base_network')
    x = Flatten(name='flatten_input')(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1, name='first_dropout')(x)
    x = Dense(128, activation='relu', name='second_base_dense')(x)
    x = Dropout(0.1, name='second_dropout')(x)
    x = Dense(128, activation='relu', name='third_base_dense')(x)
    return Model(inputs=input, outputs=x)


def euclidian_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)  # Updated to use tf.reduce_sum
    return tf.sqrt(tf.maximum(sum_square, K.epsilon()))  # Use tf.sqrt


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)




base_network = initialize_base_network()
plot_model(base_network, show_shapes=True, show_layer_names=True, to_file='TFSiameseNetworkBaseModel.png')



input_a = Input(shape=(28,28,), name="left_input")
vect_output_a = base_network(input_a)
input_b = Input(shape=(28,28,), name="right_input")
vect_output_b = base_network(input_b)

output = Lambda(euclidian_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])
model = Model([input_a, input_b], output)
plot_model(model, show_shapes=True, show_layer_names=True, to_file='TFSiameseNetworkOuterModel.png')




def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = tf.square(y_pred)  # Updated to use tf.square
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))  # Fixed typo, changed `margin_square -` to `margin_square =`
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)  # Updated to use tf.reduce_mean
    return contrastive_loss



# Compile model

rms = RMSprop()  # Ensure using tf.keras.optimizers
model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)

# Fit model
# history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, epochs=20, batch_size=128, validation_data=([ts_pairs[:, 0], ts_pairs[:, 1]]))
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, epochs=20, batch_size=128, validation_data=([ts_pairs[:, 0], ts_pairs[:, 1]], ts_y))




loss = model.evaluate(x=[ts_pairs[:,0], ts_pairs[:,1]], y=ts_y)




def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)




y_pred_train = model.predict([tr_pairs[:,0], tr_pairs[:,1]])
train_accuracy = compute_accuracy(tr_y, y_pred_train)

y_pred_test = model.predict([ts_pairs[:,0], ts_pairs[:,1]])
test_accuracy = compute_accuracy(ts_y, y_pred_test)

print("Loss = {}, Train accuracy = {} Test accuracy = {}".format(loss, train_accuracy, test_accuracy))




def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label= 'val_' + metric_name)
    plt.legend()

plot_metrics(metric_name='loss', title='Loss', ylim=0.2)    




def visualize_images():
    plt.rc('image', cmap='gray_r')
    plt.rc('grid', linewidth=0)
    plt.rc('xtick', top=False, bottom=False, labelsize='large')
    plt.rc('ytick', left=False, right=False, labelsize='large')
    plt.rc('axes', facecolor='F8F8F8', titlesize='large', edgecolor='white')  # matplotlib fonts
    plt.rc('text', color='a8151a')
    plt.rc('figure', facecolor='F0F0F0')  

# utility to display a row of digits with their prediction
def display_images(left, right, predictions, labels, title, n):
    plt.figure(figsize=(17,3))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(None)
    left = np.reshape(left, [n, 28, 28]) 
    left = np.swapaxes(left, 0, 1)
    left = np.reshape(left, [28, 28*n])
    plt.imshow(left)
    plt.figure(figsize=(17,3))
    plt.yticks([])
    plt.xticks([28*x+14 for x in range(n)], predictions)
    for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if predictions[i] > 0.5: t.set_color('red')  # bad predictions in red
    plt.grid(None)
    right = np.reshape(right, [n, 28, 28])
    right = np.swapaxes(right, 0,1)
    right = np.reshape(right, [28, 28*n])
    plt.imshow(right)



y_pred_train = np.squeeze(y_pred_train)
indexes = np.random.choice(len(y_pred_train), size=10)
display_images(tr_pairs[:, 0][indexes], tr_pairs[:,1][indexes], y_pred_train[indexes], tr_y[indexes], "clothes and their dissimilarity", 10)

