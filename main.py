import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, \
    QMessageBox, QListWidget, QComboBox, QLineEdit, QInputDialog
from PySide6.QtGui import QPainter, QMouseEvent, QImage, QColor, QPixmap, QPen
from PySide6.QtCore import Qt, QPoint
from pip._internal.utils import hashes
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import base64

import logging


# Configure logging
logging.basicConfig(
    filename='main.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Create a console handler so that logs are also displayed on the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)


#create global model variables
global mode_name
global model_list
global model
global model_accuracy
global x_test_data
global y_test_data

"""
This class is in charge of creating the canvas where the user can draw the digit. It contains the mouse event handlers
for drawing the digit, the paint event handler for rendering the drawn digit, and the clearCanvas function to clear the
canvas.
"""
class Canvas(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setFixedSize(400, 400)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() == Qt.LeftButton:
            painter = QPainter(self.image)
            pen = QPen(QColor(0, 0, 0))
            pen.setWidth(9)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def clearCanvas(self):
        self.image.fill(Qt.white)

        # Clear the cropped and resized images
        self.main_window.cropped_display.clear()
        self.main_window.resized_display.clear()
        self.main_window.predicted_label.setText("Prediction: ---")

        self.main_window.update_bar_graph([0] * 10)

        self.update()

        logging.info("Canvas cleared")

    #export the drawn digit to a numpy array
    def exportToArray(self):
        width, height = self.image.width(), self.image.height()
        pixel_array = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                color = self.image.pixelColor(x, y)
                pixel_array[y, x] = 1 if color == QColor(0, 0, 0) else 0

        return pixel_array

    """
        Crop the input pixel array to the bounding box of the digit. so that the digit is centered and padded.

        Args:
            pixel_array (np.ndarray): The input pixel array representing the drawn digit. 

        Returns:
            np.ndarray: The cropped pixel array containing the digit. 
            size of the array is proportional to the size of the digit drawn in the canvas.
        """
    def cropToBoundingBox(self, pixel_array):
        rows = np.any(pixel_array, axis=1)
        cols = np.any(pixel_array, axis=0)

        # Check if there are any non-zero pixels
        if not rows.any() or not cols.any():
            QMessageBox.warning(self.main_window, "No Drawing Detected", "Please draw a digit before marking it.")
            return np.zeros((1, 1), dtype=np.uint8)  # Return a dummy array to avoid further errors

        # Get the bounding box of the digit
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Center the digit in the bounding box
        height, width = pixel_array.shape
        box_height = y_max - y_min + 1
        box_width = x_max - x_min + 1
        padding = max(2, int(0.15 * min(box_height, box_width))) #add padding to the bounding box proportional to the size of the box(digit)

        # Make the bounding box square
        box_size = max(box_height, box_width) + 2 * padding

        # Calculate the new bounding box coordinates
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        y_min = max(0, center_y - box_size // 2)
        y_max = min(height - 1, center_y + box_size // 2)
        x_min = max(0, center_x - box_size // 2)
        x_max = min(width - 1, center_x + box_size // 2)

        # Crop the digit to the bounding box
        cropped_array = pixel_array[y_min:y_max + 1, x_min:x_max + 1]
        return cropped_array

    """
    Resize the input pixel array to 28x28 pixels using bilinear interpolation.
    
    Args:
        cropped_array (np.ndarray): The input pixel array representing the cropped digit.

    Returns:
        np.ndarray: The resized pixel array containing the digit.
        The size of the array is 28x28 pixels.    
    """
    def resizeTo28x28(self, cropped_array):
        return np.array(tf.image.resize(cropped_array[..., np.newaxis], [28, 28]).numpy() > 0.5, dtype=np.uint8).squeeze()

"""
This class is in charge of creating the main window of the application. It contains the canvas where the user can draw
the digit, the preprocessing steps, the prediction label, the clear and predict buttons, and the bar graph displaying
the class probabilities.
"""
class MainWindow(QMainWindow):
    def __init__(self):
        global model_name
        super().__init__()
        self.msg_box = None
        self.setWindowTitle("MNIST Digit Recognizer")

        self.canvas = Canvas(self)

        self.preprocess_label = QLabel("Preprocessing Steps")
        self.preprocess_label.setStyleSheet("font-size: 12px; font-weight: bold;")

        self.cropped_display = QLabel()
        self.cropped_display.setFixedSize(200, 200)
        self.cropped_display.setStyleSheet("border: 1px solid black;")

        self.resized_display = QLabel()
        self.resized_display.setFixedSize(200, 200)
        self.resized_display.setStyleSheet("border: 1px solid black;")

        self.predicted_label = QLabel("Prediction: ---")
        self.predicted_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 20px;")

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.canvas.clearCanvas)
        self.clear_button.setStyleSheet("font-size: 14px; padding: 10px;")

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.validate_and_predict)
        self.predict_button.setStyleSheet("font-size: 14px; padding: 10px;")

        self.figure, self.ax = plt.subplots()
        self.bar_canvas = FigureCanvas(self.figure)
        self.update_bar_graph([0] * 10)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.predict_button)


        #create confusion button
        self.confusion_button = QPushButton("Confusion Matrix")
        self.confusion_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.confusion_button.clicked.connect(self.show_confusion_matrix)

        #create scatter button
        self.scatter_button = QPushButton("Scatter Plot")
        self.scatter_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.scatter_button.clicked.connect(self.show_scatter_plot)


        button_layout_2 = QHBoxLayout()
        button_layout_2.addWidget(self.confusion_button)
        button_layout_2.addWidget(self.scatter_button)


        grid_layout = QHBoxLayout()

        left_grid_layout = QVBoxLayout()
        title_cropped = QLabel("1) Crop, Center and Pad")
        title_cropped.setStyleSheet("font-size: 11px;")
        left_grid_layout.addWidget(title_cropped)
        left_grid_layout.addWidget(self.cropped_display)


        right_grid_layout = QVBoxLayout()
        title_resized = QLabel("2) Resize to 28x28")
        title_resized.setStyleSheet("font-size: 11px;")
        right_grid_layout.addWidget(title_resized)
        right_grid_layout.addWidget(self.resized_display)

        self.right_grid_layout = right_grid_layout

        grid_layout.addLayout(left_grid_layout)
        grid_layout.addLayout(right_grid_layout)

        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(self.preprocess_label)
        vbox_layout.addLayout(grid_layout)
        vbox_layout.addWidget(self.bar_canvas)
        vbox_layout.addWidget(self.predicted_label)

        left_vbox_layout = QVBoxLayout()

        models_qcombo_box = QComboBox()
        #add styling to the combo box
        models_qcombo_box.setStyleSheet("font-size: 14px; padding: 10px;")


        #load the current model into the combo box
        models_qcombo_box.addItem(model_name + f" - Accuracy: {model_accuracy * 100:.2f}%")

        #for each model in the model list, add it to the combo box
        for m in model_list:
            models_qcombo_box.addItem(m)


        #onchange event for the combo box
        models_qcombo_box.currentIndexChanged.connect(self.model_changed)

        instruction_label = QLabel("Draw a digit on the white canvas below")
        instruction_label.setStyleSheet("font-size: 14px; height: 30px;")

        left_vbox_layout.addWidget(models_qcombo_box)
        left_vbox_layout.addLayout(button_layout_2)
        left_vbox_layout.addWidget(instruction_label)
        left_vbox_layout.addWidget(self.canvas)
        left_vbox_layout.addLayout(button_layout)


        main_layout = QHBoxLayout()
        main_layout.addLayout(left_vbox_layout)
        main_layout.addLayout(vbox_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        logging.info("Rendering Main Window")



    def show_confusion_matrix(self):

        predictions = model.predict(x_test_data)
        y_pred = np.argmax(predictions, axis=1)

        # Compute confusion matrix
        confusion_matrix = compute_confusion_matrix(y_test_data, y_pred, num_classes=10)

        print(confusion_matrix)

        # Plot confusion matrix
        plot_confusion_matrix(confusion_matrix, classes=[str(i) for i in range(10)])

    def show_scatter_plot(self):
        embeddings = extract_embeddings(x_test_data)
        reduced_embeddings = reduce_dimensions(embeddings)
        plot_embeddings(reduced_embeddings, y_test_data)

    def model_changed(self, i):
        global model
        global model_name
        global model_accuracy

        # if the index is 0, then the user has selected the current model
        if i == 0:
            return

        # load the selected model from the model list
        selected_model_name = model_list[i - 1]

        # prompt the user for a password to decrypt the model
        password, ok = QInputDialog.getText(self, "Password Required", "Enter password to decrypt the model:",
                                            QLineEdit.Password)
        if not ok:
            #reselect the 0 index default option
            self.sender().setCurrentIndex(0)
            return
        if not password:
            QMessageBox.warning(self, "Password Required", "Please enter a decryption password to continue.")
            return



        load_or_train_model(selected_model_name, password)

        #if model is no defined, display an error message
        if model is None:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText(f"Failed to load the selected model. Please make sure the password is correct.")
            msg_box.exec()
            return

        # reload main window
        self.close()
        main_window = MainWindow()
        main_window.show()



    def update_bar_graph(self, probabilities):
        # Clear the existing figure and axes
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        self.ax.clear()
        self.ax.bar(range(10), probabilities, color='blue')
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel("Digits")
        self.ax.set_ylabel("Probability")
        self.ax.set_title("Digit Probabilities")
        self.bar_canvas.draw()

    """
    This function is in charge of validating the input drawn by the user and predicting the digit using the CNN model.
    
    Returns:
        None
    
    """
    def validate_and_predict(self):
        logging.info("Validating and predicting digit")
        pixel_array = self.canvas.exportToArray()

        labeled_array, num_features = label(pixel_array)

        min_pixel_threshold = 500
        valid_clusters = 0

        #check if there are any valid clusters that meet the min pixel threshold
        for i in range(1, num_features + 1):
            cluster_size = np.sum(labeled_array == i)
            if cluster_size >= min_pixel_threshold:
                valid_clusters += 1

        #extract the number of pixels in each cluster
        cluster_info = ", ".join([f"{i}) {np.sum(labeled_array == i)} pixels" for i in range(1, num_features + 1)])
        logging.info(f"Pixel cluster sizes: {cluster_info}")

        #remove clusters that do not meet the min pixel threshold
        for i in range(1, num_features + 1):
            cluster_size = np.sum(labeled_array == i)
            if cluster_size < min_pixel_threshold:
                pixel_array[labeled_array == i] = 0

        #check if there are multiple drawings (more than one separate cluster of pixels)
        if valid_clusters > 1:
            logging.warning("Multiple drawings detected")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Please draw only one digit at a time to ensure the CNN works accurately. \n Use the clear button to clear the canvas.")
            msg_box.exec()
            return

        self.predict_digit(pixel_array)

    """
    This function is in charge of predicting the digit using the CNN model.
    
    Args:
        pixel_array (np.ndarray): The input pixel array representing the drawn digit
    
    Returns:
        None
    
    """
    def predict_digit(self, pixel_array):
        global model

        #check if the pixel array is empty
        if np.sum(pixel_array) == 0:
            logging.warning("No drawing detected")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Please draw a digit before predicting.")
            msg_box.exec()
            return

        # Crop, center, and pad the digit
        cropped_array = self.canvas.cropToBoundingBox(pixel_array)

        cropped_height, cropped_width = cropped_array.shape
        cropped_image = QImage(cropped_width, cropped_height, QImage.Format_RGB32)
        cropped_image.fill(Qt.white)

        for y in range(cropped_height):
            for x in range(cropped_width):
                if cropped_array[y, x] == 1:
                    cropped_image.setPixelColor(x, y, QColor(0, 0, 0)) #set the pixel color to black

        #display the cropped image
        cropped_pixmap = QPixmap.fromImage(cropped_image).scaled(
            self.cropped_display.width(),
            self.cropped_display.height(),
            Qt.KeepAspectRatio
        )
        self.cropped_display.setPixmap(cropped_pixmap)

        # Resize the digit to 28x28 pixels
        resized_array = self.canvas.resizeTo28x28(cropped_array)

        resized_image = QImage(28, 28, QImage.Format_RGB32)
        resized_image.fill(Qt.white)

        #set the pixel colors to black
        for y in range(28):
            for x in range(28):
                if resized_array[y, x] == 1:
                    resized_image.setPixelColor(x, y, QColor(0, 0, 0))

        #display the resized image
        resized_pixmap = QPixmap.fromImage(resized_image).scaled(
            self.resized_display.width(),
            self.resized_display.height(),
            Qt.KeepAspectRatio
        )
        self.resized_display.setPixmap(resized_pixmap)

        # Prepare the digit array for prediction
        digit_array = resized_array.astype('float32').reshape(1, 28, 28, 1)

        # Predict the digit using the CNN model
        prediction = model.predict(digit_array)
        predicted_digit = np.argmax(prediction)

        self.update_bar_graph(prediction[0])

        logging.info(f"Predicted digit: {predicted_digit}")
        logging.info("Digit probabilities: " + ", ".join([f"{i}: {prob:.4f}" for i, prob in enumerate(prediction[0])]))

        self.predicted_label.setText(f"Prediction: {predicted_digit}")



'''
This window is in charge of reading the contents of the /models directory and displaying the available models in a list 
for the user to select. The models are displayed in a dropdown list, in which the default option is to train a new model.
the name of the new model will be "mnist_model_v_X.keras" where X is the next available number in the sequence.
The "continue" button allows the user to continue with or without selecting a model.
'''
class ModelSelectionWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CNN Model Selection")
        self.setGeometry(100, 100, 400, 400)

        self.model_list_label = QLabel("Select a saved model or train a new one:")
        self.model_list_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.model_list_widget = QListWidget()
        self.model_list_widget.setStyleSheet("font-size: 12px;")
        self.model_list_widget.addItem("+ Train a new model")

        self.model_list_widget.setStyleSheet("font-size: 14px; padding: 2px;")

        for m in model_list:
            self.model_list_widget.addItem(m)

        #add password field so that the user can enter the password to decrypt the model
        self.password_field = QLineEdit()
        self.password_field.setPlaceholderText("Enter password to decrypt model")
        self.password_field.setEchoMode(QLineEdit.Password)



        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_button_clicked)
        self.continue_button.setStyleSheet("font-size: 14px; padding: 10px;")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.model_list_label)
        self.layout.addWidget(self.model_list_widget)
        self.layout.addWidget(self.password_field)
        self.layout.addWidget(self.continue_button)

        self.setLayout(self.layout)
        self.show()
        self.model_selection_window = self

        #select the first item in the list by default
        self.model_list_widget.setCurrentRow(0)

        logging.info("Rendering Model Selection Window")

    """
    This function is in charge of handling the continue button click event. It validates the selected model and loads it and closes the model selection window.
    
    """
    def continue_button_clicked(self):
        global model

        #if password is not entered, display an error message
        if self.password_field.text() == "":
            logging.warning("No password entered")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)

            #if + Train a new model is selected, display a different message
            if self.model_list_widget.currentItem().text() == "+ Train a new model":
                msg_box.setText(f"Please enter a password to encrypt the new model.")
            else:
                msg_box.setText(f"Please enter the password to decrypt the model.")

            msg_box.exec()
            return

        # Get the selected model
        selected_model_name = self.model_list_widget.currentItem().text()
        logging.info(f"Selected Option: {selected_model_name}")

        #verify that the extension is .keras and is not the default option
        if selected_model_name != "+ Train a new model" and selected_model_name[-6:] != ".keras":
            logging.warning("Invalid model file selected")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Please select a keras model to continue.")
            msg_box.exec()
            return

        password = self.password_field.text()

        #if + Train a new model is selected, confirm the password with a dialog box
        if selected_model_name == "+ Train a new model":
            password, ok = QInputDialog.getText(self, "Confirm Password", "Please confirm the password to encrypt the new model:",
                                                QLineEdit.Password)
            if not ok:
                return
            if not password:
                QMessageBox.warning(self, "Password Required", "Please enter a password to encrypt the new model.")
                return

        load_or_train_model(selected_model_name, password)

        #if model is no defined, display an error message
        if model is None:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText(f"Failed to load the selected model. Please make sure the password is correct.")
            msg_box.exec()
            return

        self.model_selection_window.close()
        main_window = MainWindow()
        main_window.show()

"""
This function is in charge of loading the selected model or training a new model if the user selects the option to train a new model.

"""
def load_or_train_model(model_file, password):
    global model
    global model_name
    global model_accuracy
    global x_test_data
    global y_test_data

    # Load MNIST dataset and preprocess
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)

    x_test_data = x_test
    y_test_data = y_test

    if model_file == "+ Train a new model":

        model_file = f"models/mnist_model_v_{len(model_list) + 1}.keras"

        #extract the model name from the file path
        model_name = model_file.split("/")[1]
        logging.info("Training new model: " + model_name)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)


        # Define the path for the encrypted file
        encrypted_file_path = os.path.join("models", f"encrypted_{len(model_list) + 1}.keras")

        # Train and save the new model temporarily
        model.save("temp_model.keras")

        # Encrypt the model and save it in the models folder
        encrypt_file("temp_model.keras", encrypted_file_path, password)

        # Remove the temporary plain-text model file
        os.remove("temp_model.keras")

        logging.info("Model successfully saved and loaded: " + model_file)
    else:

        file_name = f"models/{model_file}"

        temp_model_file = "temp_model.keras"
        try:
            decrypt_file(file_name, temp_model_file, password)
        except Exception as e:
            logging.error(f"Failed to decrypt model: {e}")
            model = None
            return
        model = load_model(temp_model_file)
        os.remove(temp_model_file)
        logging.info("Model successfully decrypted and loaded.")

        # remove the models/ prefix from the model name
        model_name = model_file.split("/")[0]

        logging.info("Model successfully loaded: " + model_name)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0, to_categorical(y_test, 10))
    model_accuracy = accuracy
    logging.info(f"Model accuracy: {accuracy:.4f}")

def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[true_label, predicted_label] += 1

    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, classes):

    # clear the current plot
    plt.clf()
    plt.close()


    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: " + model_name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Display the values in the confusion matrix
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2. else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

def extract_embeddings(x_data):
    global model

    # Use the penultimate layer's output as embeddings
    embedding_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.layers[-2].output
    )

    embeddings = embedding_model.predict(x_data)
    return embeddings

def reduce_dimensions(embeddings, num_dimensions=2):
    # Simple PCA for dimensionality reduction
    mean = np.mean(embeddings, axis=0)
    embeddings_centered = embeddings - mean
    covariance_matrix = np.cov(embeddings_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_dimensions]]
    reduced_embeddings = np.dot(embeddings_centered, top_eigenvectors)
    return reduced_embeddings

def plot_embeddings(embeddings, labels):
    # clear the current plot
    plt.clf()
    plt.close()

    plt.figure(figsize=(10, 10))

    # Plot embeddings with colors corresponding to labels
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Classes')

    plt.title("Scatterplot of Embeddings: " + model_name)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.tight_layout()
    plt.show()


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a symmetric encryption key from a password."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())


def encrypt_file(input_file: str, output_file: str, password: str):
    """Encrypt the model file using AES encryption."""
    salt = os.urandom(16)
    key = derive_key(password, salt)
    with open(input_file, "rb") as f:
        data = f.read()

    cipher = Cipher(algorithms.AES(key), modes.GCM(salt), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data) + encryptor.finalize()

    with open(output_file, "wb") as f:
        f.write(salt + encryptor.tag + encrypted_data)


def decrypt_file(input_file: str, output_file: str, password: str):
    """Decrypt the model file using AES decryption."""
    with open(input_file, "rb") as f:
        data = f.read()

    salt = data[:16]
    tag = data[16:32]
    encrypted_data = data[32:]

    key = derive_key(password, salt)
    cipher = Cipher(algorithms.AES(key), modes.GCM(salt, tag), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

    with open(output_file, "wb") as f:
        f.write(decrypted_data)


if __name__ == "__main__":
    global model_list

    try:
        app = QApplication(sys.argv)
        logging.info("Starting application")

        # get list of models from /models directory
        model_list = os.listdir("models")

        # log the models found in the directory
        logging.info(f"Models found: {model_list}")

        model_selection_window = ModelSelectionWindow()

        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit()
