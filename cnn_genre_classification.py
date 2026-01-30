import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras 

DATASET_PATH = "data.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y

def prepare_dataset(test_size, val_size):
    X, y = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val   =  train_test_split(X_train, y_train, test_size=val_size)

    X_train = X_train[..., np.newaxis] #4d array -> (num_sampels, num_bins, nums_mfcc, depth)
    X_val = X_val[..., np.newaxis] #4d array
    X_test = X_test[..., np.newaxis] #4d array

    return  X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2,2), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

def predict(model, X, y):
    X = X[np.newaxis, ...]

    prediction = model.predict(X)

    predicted_index = np.argmax(prediction, axis=1)

    print("expected index: {}, predixted index: {}".format(y, predicted_index))

if __name__ == "__main__":

    X, y = load_data(DATASET_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(0.25, 0.2)

    input_shape = (X_train.shape[1],  X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["Accuracy"])
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=30)

    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print("accuracy in test set: {}".format(test_accuracy))

    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)