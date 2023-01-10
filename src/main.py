# Importing Frameworks We Need for the Algorithm.
import cv2
import keras
import numpy as np
import tensorflow as tf
from random import randint
from tensorflow import keras
import matplotlib.pyplot as plt

def mnist():
    # Gathering models with mnist.
    # Splitting the models between test and train.
    (x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data() 

    # Scaling our data to range [0,1]
    x_train = x_train / 255
    x_test = x_test / 255

    # Flatting the test data
    x_train_flattened = x_train.reshape(len(x_train), 28*28)
    x_test_flattened = x_test.reshape(len(x_test), 28*28)

    # Creating our neural network.
    model = keras.Sequential([
        keras.layers.Dense(units = 128, input_shape=(784,), activation='sigmoid'),
        keras.layers.Dense(units = 128, input_shape=(784,), activation='sigmoid')
    ])

    # Compiling the whole thing. 
    # Selecting > Optimizer Function, Loss Funciton and Metrics.
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # Fitting the train and test data.
    model.fit(x_train_flattened, y_train, epochs=10)

    loss, accuracy = model.evaluate(x_test_flattened, y_test)

    print(f"\nAverage loss is: {loss}") # To show loss we got.
    print(f"Accurary is: {accuracy}\n") # To show accuracy we got.

    # Picking a random handwrite digit to predict.
    randomm = randint(0, 99)

    # Realizing the real time photo of digit prediction.
    y_predicted = model.predict(x_test_flattened)
    plt.title(f"\nPrediction is: {np.argmax(y_predicted[randomm])}")
    plt.imshow(x_test[randomm])
    plt.show()

    askk = input("Would you like to save the model (y/n)?: ")
    if askk == "y":
        model.save('digits.model') # Saving the model but not necessary.
    elif askk == "n":
        pass
    else:
        print("Wrong choice, Model couldn't saved.")

def image(nameimage):
    # Gathering models with mnist.
    # Splitting the models between test and train.
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Scaling our data.
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Creating our neural network. Defining the epoch.
    e = 10
    # Using two hidden layers and each of them has 16 neurons.
    # Added one flattened final input for pixel recognition.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling the whole thing. 
    # Selecting > Optimizer Function, Loss Funciton and Metrics.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fitting the train and test data and analyzing model history.
    history = model.fit(x_train, y_train, batch_size=64, validation_data=(x_test, y_test), epochs=e)
    print(history)
    
    # Summarizing our models statistics.
    model.summary()

    # Evaluating the model for the final result.
    model.evaluate(x_test, y_test)

    # loading custom images to predict them.
    try:
        img = cv2.imread(nameimage)[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        plt.title(f"\nThe number is probably {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print(f"\nThere is no image named {nameimage}.")

    askk = input("\nWould you like to save the model (y/n)?: ")
    if askk == "y":
        # to save the model.
        model.save('digits.model')
    elif askk == "n":
        pass
    else:
        print("Wrong choice, model couldn't saved.")
