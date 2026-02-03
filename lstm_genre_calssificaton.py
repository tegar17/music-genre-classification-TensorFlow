import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras 
import matplotlib.pyplot as plt

DATASET_PATH = "data.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y

def plot_history(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label = "train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("accuracy eval")

    axs[1].plot(history.history["loss"], label = "train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("error")
    axs[1].legend(loc="lower right")
    axs[1].set_title("error eval")

    plt.show()

def prepare_dataset(test_size, val_size):
    X, y = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val   =  train_test_split(X_train, y_train, test_size=val_size)

    return  X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape):
    model = keras.Sequential()

    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

if __name__ == "__main__":

    X, y = load_data(DATASET_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(0.25, 0.2)

    input_shape = (X_train.shape[1],  X_train.shape[2])
    model = build_model(input_shape)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=32)
    plot_history(history=history)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)