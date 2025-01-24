import os
import tensorflow as tf
from PySide6.QtWidgets import QMessageBox
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import logging
import log_config  # Custom logging configuration
import state  # Global application state
from encryption import decrypt_file, encrypt_file

def load_mnist_data():
    """Load and preprocess the MNIST dataset."""
    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Normalize pixel values to the range [0, 1] and reshape for CNN
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0 #convert to float32 28x28x1 (the 1 is for
        y_train = to_categorical(y_train, 10)

        state.x_train_data = x_train
        state.y_train_data = y_train
        state.x_test_data = x_test
        state.y_test_data = y_test
        logging.info("MNIST dataset loaded and preprocessed.")
    except Exception as e:
        logging.error(f"Error loading MNIST data: {e}")

def load_cnn_model(model_file, password):
    """Decrypt and load a pre-trained CNN model."""
    file_path = f"models/{model_file}"
    temp_model_file = "temp_model.keras"

    try:
        # Decrypt the model file
        decrypt_file(file_path, temp_model_file, password)
        state.model = load_model(temp_model_file)
        os.remove(temp_model_file)

        # Store the model name in the global state
        state.model_name = os.path.basename(model_file).split(".")[0]
        logging.info(f"Model {state.model_name} successfully loaded.")
    except Exception as e:
        logging.error(f"Failed to decrypt and load model: {e}")
        show_error_message("Failed to decrypt model. Please ensure the password is correct.")

def train_new_model(password):
    """Train a new CNN model on the MNIST dataset and save it encrypted."""
    try:
        model_name = f"mnist_model_v_{len(state.model_list) + 1}"
        model_file = f"models/{model_name}.keras"

        logging.info(f"Starting training for {model_name}.")

        # Define the CNN architecture
        state.model = tf.keras.Sequential([

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')

        ])

        # Compile the model
        state.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        state.model.fit(state.x_train_data, state.y_train_data, epochs=5, batch_size=128, validation_split=0.1)

        # Save and encrypt the model
        state.model.save("temp_model.keras")
        encrypt_file("temp_model.keras", model_file, password)
        os.remove("temp_model.keras")

        logging.info(f"Model {model_name} successfully trained and saved.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")

def evaluate_model_accuracy():
    """Evaluate the model's accuracy on the test dataset."""
    try:
        # Normalize and reshape test data
        x_test = state.x_test_data.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_test = to_categorical(state.y_test_data, 10)

        loss, accuracy = state.model.evaluate(x_test, y_test)
        state.model_accuracy = accuracy
        logging.info(f"Model accuracy: {accuracy:.4f}")
    except Exception as e:
        logging.error(f"Error evaluating model accuracy: {e}")

def show_error_message(message):
    """Display an error message using QMessageBox."""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText(message)
    msg_box.exec()

if __name__ == "__main__":
    import main
    main.main()
