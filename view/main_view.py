import os

import numpy as np
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, \
    QMessageBox, QComboBox, QLineEdit, QInputDialog
from PySide6.QtGui import QPainter, QImage, QColor, QPixmap, QPen
from PySide6.QtCore import Qt, QPoint
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import tensorflow as tf


import logging
import log_config
import state
from cnn import load_cnn_model, evaluate_model_accuracy
from visuals import compute_confusion_matrix, plot_confusion_matrix, extract_embeddings, reduce_dimensions, \
    plot_embeddings


class Canvas(QWidget):
    """Canvas widget for drawing digits."""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setFixedSize(400, 400)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
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
        """Clear the canvas."""
        self.image.fill(Qt.white)

        # Clear the cropped and resized images
        self.main_window.cropped_display.clear()
        self.main_window.resized_display.clear()
        self.main_window.predicted_label.setText("Prediction: ---")

        self.main_window.update_bar_graph([0] * 10)

        self.update()

        logging.info("Canvas cleared")

    # export the drawn digit to a numpy array
    def exportToArray(self):
        """Export the drawn digit to a numpy array."""
        width, height = self.image.width(), self.image.height()
        pixel_array = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                color = self.image.pixelColor(x, y)
                pixel_array[y, x] = 1 if color == QColor(0, 0, 0) else 0

        return pixel_array


    def cropToBoundingBox(self, pixel_array):
        """
        Crop the input pixel array to the bounding box of the digit.
        Args:
            pixel_array (np.ndarray): The input pixel array representing the drawn digit.

        Returns:
            np.ndarray: The cropped pixel array containing the digit.
        """
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
        padding = max(2, int(0.15 * min(box_height,
                                        box_width)))  # add padding to the bounding box proportional to the size of the box(digit)

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


    def resizeTo28x28(self, cropped_array):
        """
        Resize the cropped digit to 28x28 pixels.
        Args:
            cropped_array (np.ndarray): The cropped pixel array containing the digit.

        Returns:
            np.ndarray: The resized pixel array of the digit.
        """
        return np.array(tf.image.resize(cropped_array[..., np.newaxis], [28, 28]).numpy() > 0.5,
                        dtype=np.uint8).squeeze()



class MainWindow(QMainWindow):
    """
    Main application window for the MNIST Digit Recognizer.
    """
    def __init__(self):

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

        # create confusion button
        self.confusion_button = QPushButton("Confusion Matrix")
        self.confusion_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.confusion_button.clicked.connect(self.show_confusion_matrix)

        # create scatter button
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
        # add styling to the combo box
        models_qcombo_box.setStyleSheet("font-size: 14px; padding: 10px;")

        # load the current model into the combo box
        models_qcombo_box.addItem(state.model_name + f" - Accuracy: {state.model_accuracy * 100:.2f}%")

        #reload the model list
        state.model_list = os.listdir("models")

        # for each model in the model list, add it to the combo box
        for m in state.model_list:
            models_qcombo_box.addItem(m)

        # onchange event for the combo box
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

        predictions = state.model.predict(state.x_test_data)
        y_pred = np.argmax(predictions, axis=1)

        # Compute confusion matrix
        confusion_matrix = compute_confusion_matrix(state.y_test_data, y_pred, num_classes=10)

        # Plot confusion matrix
        plot_confusion_matrix(confusion_matrix, classes=[str(i) for i in range(10)])

    def show_scatter_plot(self):
        embeddings = extract_embeddings(state.x_test_data)
        reduced_embeddings = reduce_dimensions(embeddings)
        plot_embeddings(reduced_embeddings, state.y_test_data)

    def model_changed(self, i):

        # if the index is 0, then the user has selected the current model
        if i == 0:
            return

        # load the selected model from the model list
        selected_model_name = state.model_list[i - 1]

        # prompt the user for a password to decrypt the model
        password, ok = QInputDialog.getText(self, "Password Required", "Enter password to decrypt the model:",
                                            QLineEdit.Password)
        if not ok:
            # reselect the 0 index default option
            self.sender().setCurrentIndex(0)
            return
        if not password:
            QMessageBox.warning(self, "Password Required", "Please enter a decryption password to continue.")
            # reselect the 0 index default option
            self.sender().setCurrentIndex(0)
            return

        load_cnn_model(selected_model_name, password)

        #reevaluate the model accuracy
        evaluate_model_accuracy()

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

        # check if there are any valid clusters that meet the min pixel threshold
        for i in range(1, num_features + 1):
            cluster_size = np.sum(labeled_array == i)
            if cluster_size >= min_pixel_threshold:
                valid_clusters += 1

        # extract the number of pixels in each cluster
        cluster_info = ", ".join([f"{i}) {np.sum(labeled_array == i)} pixels" for i in range(1, num_features + 1)])
        logging.info(f"Pixel cluster sizes: {cluster_info}")

        # remove clusters that do not meet the min pixel threshold
        for i in range(1, num_features + 1):
            cluster_size = np.sum(labeled_array == i)
            if cluster_size < min_pixel_threshold:
                pixel_array[labeled_array == i] = 0

        # check if there are multiple drawings (more than one separate cluster of pixels)
        if valid_clusters > 1:
            logging.warning("Multiple drawings detected")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(
                f"Please draw only one digit at a time to ensure the CNN works accurately. \n Use the clear button to clear the canvas.")
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

        # check if the pixel array is empty
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
                    cropped_image.setPixelColor(x, y, QColor(0, 0, 0))  # set the pixel color to black

        # display the cropped image
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

        # set the pixel colors to black
        for y in range(28):
            for x in range(28):
                if resized_array[y, x] == 1:
                    resized_image.setPixelColor(x, y, QColor(0, 0, 0))

        # display the resized image
        resized_pixmap = QPixmap.fromImage(resized_image).scaled(
            self.resized_display.width(),
            self.resized_display.height(),
            Qt.KeepAspectRatio
        )
        self.resized_display.setPixmap(resized_pixmap)

        # Prepare the digit array for prediction
        digit_array = resized_array.astype('float32').reshape(1, 28, 28, 1)

        # Predict the digit using the CNN model
        prediction = state.model.predict(digit_array)
        predicted_digit = np.argmax(prediction)

        self.update_bar_graph(prediction[0])

        logging.info(f"Predicted digit: {predicted_digit}")
        logging.info("Digit probabilities: " + ", ".join([f"{i}: {prob:.4f}" for i, prob in enumerate(prediction[0])]))

        self.predicted_label.setText(f"Prediction: {predicted_digit}")


if __name__ == "__main__":
    print("Please run main.py to start the application.")