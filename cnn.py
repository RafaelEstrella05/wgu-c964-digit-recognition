import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
#---

import logging
import log_config # log_config.py
import state
from encryption import decrypt_file, encrypt_file


def load_mnist_data():
    # Load MNIST dataset and preprocess
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)

    state.x_train_data = x_train
    state.y_train_data = y_train
    state.x_test_data = x_test
    state.y_test_data = y_test


def load_cnn_model(model_file, password):

    file_name = f"models/{model_file}"

    temp_model_file = "temp_model.keras"
    try:
        decrypt_file(file_name, temp_model_file, password)
    except Exception as e:
        logging.error(f"Failed to decrypt model: {e}")
        state.model = None
        return

    state.model = load_model(temp_model_file)
    os.remove(temp_model_file)

    # remove the models/ prefix from the model name
    state.model_name = model_file.split("/")[0]
    logging.info("Model successfully decrypted and loaded")


def train_new_model(password):
    model_file = f"models/mnist_model_v_{len(state.model_list) + 1}.keras"

    # extract the model name from the file path
    state.model_name = model_file.split("/")[1]
    logging.info("Training new model: " + state.model_name)

    state.model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    state.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    state.model.fit(state.x_train_data, state.y_train_data, epochs=5, batch_size=128, validation_split=0.1)

    # Define the path for the encrypted file
    encrypted_file_path = os.path.join("models", f"encrypted_{len(state.model_list) + 1}.keras")

    # Train and save the new model temporarily
    state.model.save("temp_model.keras")

    # Encrypt the model and save it in the models folder
    encrypt_file("temp_model.keras", encrypted_file_path, password)

    # Remove the temporary plain-text model file
    os.remove("temp_model.keras")

    logging.info("Model successfully saved and loaded: " + model_file)


def evaluate_model_accuracy():
    # Evaluate the model on the test data
    loss, accuracy = state.model.evaluate(state.x_test_data.reshape(-1, 28, 28, 1).astype('float32') / 255.0,
                                          to_categorical(state.y_test_data, 10))
    state.model_accuracy = accuracy
    logging.info(f"Model accuracy: {accuracy:.4f}")