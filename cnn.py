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
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Normalize pixel values of x_train to the range [0, 1] and reshape for CNN
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
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

def train_new_model_v1(model_name, password):
    """Train a basic CNN model with minimal configuration."""

    """
        This model configuration includes a single convolutional layer, a pooling layer, and two dense layers.
        It utilizes the SGD optimizer with a learning rate of 0.01 and a categorical cross-entropy loss function.
        The model is trained with a batch size of 128 for 3 epochs, using a validation split of 0.1.
    """
    try:
        logging.info(f"Starting training for {model_name} - Version 1.")

        # Version 1: Simple model with one convolutional and pooling layer
        state.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        state.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        state.model.fit(state.x_train_data, state.y_train_data, epochs=3, batch_size=128, validation_split=0.1)
        save_and_encrypt_model(model_name, password)

        logging.info("Version 1 completed: Basic architecture with SGD optimizer.")
    except Exception as e:
        logging.error(f"Error during model training (Version 1): {e}")


def train_new_model_v2(model_name, password):
    """Train a CNN model with increased complexity and improved optimizer."""
    """
        This model configuration includes an additional convolutional layer and a switch to the Adam optimizer.
        The model architecture consists of two convolutional layers, two pooling layers, and two dense layers.
        The model is trained with a batch size of 128 for 5 epochs, using a validation split of 0.1.
        
        Key updates from Version 1:
        - Additional convolutional layer
        - Switch to Adam optimizer
        - More epochs for training
        
    """
    try:
        logging.info(f"Starting training for {model_name} - Version 2.")

        # Version 2: Add an additional convolutional layer and switch to Adam optimizer
        state.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        state.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        state.model.fit(state.x_train_data, state.y_train_data, epochs=5, batch_size=128, validation_split=0.1)
        save_and_encrypt_model(model_name, password)

        logging.info("Version 2 completed: Additional convolutional layer and Adam optimizer.")
    except Exception as e:
        logging.error(f"Error during model training (Version 2): {e}")


def train_new_model_v3(model_name, password):
    """Train a CNN model with dropout and larger dense layers."""

    """
        This model configuration introduces dropout layers to reduce overfitting and increase generalization.
        The model architecture consists of two convolutional layers, two pooling layers, and two dense layers with dropout.
        The model is trained with a batch size of 64 for 8 epochs, using a validation split of 0.1.
        
        Key updates from Version 2:
        - Dropout layers added to reduce overfitting
        - Larger dense layer (128 units)
        - More epochs for training
        
        Expected Observation Confusion Matrix and Embeddings Scatterplot
        - The model should perform better on the test dataset due to the dropout layers.
        - The confusion matrix should show improved performance across all classes. 
    """

    try:
        logging.info(f"Starting training for {model_name} - Version 3.")

        # Version 3: Introduce dropout layers to reduce overfitting
        state.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        state.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        state.model.fit(state.x_train_data, state.y_train_data, epochs=8, batch_size=64, validation_split=0.1)
        save_and_encrypt_model(model_name, password)

        logging.info("Version 3 completed: Dropout layers added to reduce overfitting.")
    except Exception as e:
        logging.error(f"Error during model training (Version 3): {e}")

def train_new_model(model_name, password):
    #train_new_model_v1(model_name, password) # Version 1: 91.83% accuracy
    #train_new_model_v2(model_name, password) # Version 2: 98.87% accuracy
    train_new_model_v3(model_name, password) # Version 3: 99.25%

    #update model name
    state.model_name = model_name
    evaluate_model_accuracy()
    logging.info("Model training completed successfully.")
    state.model_list.append(model_name)

def save_and_encrypt_model(model_name, password):
    """Helper function to save and encrypt the trained model."""
    try:
        model_file = f"models/{model_name}.keras"
        state.model.save("temp_model.keras")
        encrypt_file("temp_model.keras", model_file, password)
        os.remove("temp_model.keras")
        logging.info(f"Model {model_name} successfully saved and encrypted.")
    except Exception as e:
        logging.error(f"Error saving and encrypting model: {e}")


def evaluate_model_accuracy():
    """Evaluate the model's accuracy on the test dataset."""
    try:
        # Normalize and reshape test data
        x_test = state.x_test_data.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_test = to_categorical(state.y_test_data, 10)

        # Evaluate the model on the test data
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
