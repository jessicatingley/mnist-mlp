import numpy as np
import random
import tensorflow as tf
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.optimizers import SGD, RMSprop, Adam, Adagrad


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    #Prepare training dataset
    train_images = train_images.reshape(60000, 28*28)
    train_images = train_images.astype('float32') / 255.0

    #Prepare test dataset
    test_images = test_images.reshape(10000, 28*28)
    test_images = test_images.astype('float32') / 255.0

    #Prepare labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    #Split dataset
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    # print("Train set shapes:\nImages:", train_images.shape, "\nLabels:", train_labels.shape, "\n")
    # print("Validation set shapes:\nImages:", val_images.shape, "\nLabels:", val_labels.shape, "\n")
    # print("Test set shapes:\nImages:", test_images.shape, "\nLabels:", test_labels.shape, "\n")

    return (train_images, train_labels, test_images, test_labels, val_images, val_labels)


def create_model(num_neurons: int, act_func: str, num_layers: int, dropout: float, learn_rate: float, optimizer_choice: str):
    model = models.Sequential()

    for i in range(num_layers):
        model.add(layers.Dense(num_neurons, activation=act_func, input_shape=(28*28,)))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(10, activation='softmax'))

    choice_to_opt = {"rmsprop": RMSprop, "sgd" : SGD, "adam" : Adam, "adagrad" : Adagrad}
    optimizer = choice_to_opt[optimizer_choice](learning_rate=learn_rate)
    model.compile(optimizer=optimizer, 
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def train_model(model, train_images: np.array, train_labels: np.array, epochs: int, batch_size: int):
    hist = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    return hist.history['loss'], hist.history['accuracy']


def evaluate_model(model, images: np.array, labels: np.array):
    loss, acc = model.evaluate(images, labels)
    return loss, acc
