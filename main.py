import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PySide6.QtGui import QPainter, QMouseEvent, QImage, QColor, QPixmap, QPen
from PySide6.QtCore import Qt, QPoint
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# File name for saving and loading the trained model
model_file = "mnist_model.keras"

# Load the MNIST dataset and preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Load or train model
if os.path.exists(model_file):
    model = load_model(model_file)
else:
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
    model.save(model_file)

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

    def exportToArray(self):
        width, height = self.image.width(), self.image.height()
        pixel_array = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                color = self.image.pixelColor(x, y)
                pixel_array[y, x] = 1 if color == QColor(0, 0, 0) else 0

        return pixel_array

    def cropToBoundingBox(self, pixel_array):
        rows = np.any(pixel_array, axis=1)
        cols = np.any(pixel_array, axis=0)

        # Check if there are any non-zero pixels
        if not rows.any() or not cols.any():
            QMessageBox.warning(self.main_window, "No Drawing Detected", "Please draw a digit before marking it.")
            return np.zeros((1, 1), dtype=np.uint8)  # Return a dummy array to avoid further errors

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        height, width = pixel_array.shape
        box_height = y_max - y_min + 1
        box_width = x_max - x_min + 1
        padding = max(2, int(0.15 * min(box_height, box_width)))

        box_size = max(box_height, box_width) + 2 * padding

        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        y_min = max(0, center_y - box_size // 2)
        y_max = min(height - 1, center_y + box_size // 2)
        x_min = max(0, center_x - box_size // 2)
        x_max = min(width - 1, center_x + box_size // 2)

        cropped_array = pixel_array[y_min:y_max + 1, x_min:x_max + 1]
        return cropped_array

    def resizeTo28x28(self, cropped_array):
        return np.array(tf.image.resize(cropped_array[..., np.newaxis], [28, 28]).numpy() > 0.5, dtype=np.uint8).squeeze()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Digit Recognizer")

        self.canvas = Canvas(self)

        self.title_label = QLabel("MNIST Digit Recognizer")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

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

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.validate_and_predict)

        self.figure, self.ax = plt.subplots()
        self.bar_canvas = FigureCanvas(self.figure)
        self.update_bar_graph([0] * 10)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.predict_button)

        grid_layout = QHBoxLayout()

        left_grid_layout = QVBoxLayout()
        title_cropped = QLabel("1) Crop, Center and Pad")
        title_cropped.setStyleSheet("font-size: 11px;")
        left_grid_layout.addWidget(title_cropped)
        left_grid_layout.addWidget(self.cropped_display)

        incorrect_button = QPushButton("Correction")
        incorrect_button.clicked.connect(self.mark_incorrect)
        left_grid_layout.addWidget(incorrect_button)

        right_grid_layout = QVBoxLayout()
        title_resized = QLabel("2) Resize to 28x28")
        title_resized.setStyleSheet("font-size: 11px;")
        right_grid_layout.addWidget(title_resized)
        right_grid_layout.addWidget(self.resized_display)

        grid_layout.addLayout(left_grid_layout)
        grid_layout.addLayout(right_grid_layout)

        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(self.title_label)
        vbox_layout.addWidget(self.preprocess_label)
        vbox_layout.addLayout(grid_layout)
        vbox_layout.addWidget(self.bar_canvas)
        vbox_layout.addWidget(self.predicted_label)
        vbox_layout.addLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(vbox_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def mark_incorrect(self):
        # Create a dialog for digit selection
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Mark Incorrect")
        msg_box.setText("Select the correct digit:")
        msg_box.setStandardButtons(QMessageBox.NoButton)  # No default buttons

        # Create custom buttons for digits
        for i in range(10):
            button = QPushButton(str(i), msg_box)
            button.clicked.connect(lambda _, digit=i: self.save_corrected_data(digit, msg_box))
            msg_box.layout().addWidget(button)

        msg_box.exec()

    def save_corrected_data(self, correct_digit, msg_box):
        # Process the corrected data and save it
        resized_array = self.canvas.resizeTo28x28(self.canvas.cropToBoundingBox(self.canvas.exportToArray()))
        normalized_image = resized_array.astype('float32') / 255.0

        save_path = "user_corrected_data.csv"
        with open(save_path, "a") as f:
            flattened_data = [correct_digit] + normalized_image.flatten().tolist()
            f.write(",".join(map(str, flattened_data)) + "\n")

        self.canvas.clearCanvas()  # Clear the canvas after saving
        msg_box.close()

    def update_bar_graph(self, probabilities):
        self.ax.clear()
        self.ax.bar(range(10), probabilities, color='blue')
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel("Digits")
        self.ax.set_ylabel("Probability")
        self.ax.set_title("Class Probabilities")
        self.bar_canvas.draw()

    def validate_and_predict(self):
        pixel_array = self.canvas.exportToArray()

        labeled_array, num_features = label(pixel_array)

        min_pixel_threshold = 500
        valid_clusters = 0

        for i in range(1, num_features + 1):
            cluster_size = np.sum(labeled_array == i)
            if cluster_size >= min_pixel_threshold:
                valid_clusters += 1

        print("Analyzing Pixel Clusters")
        cluster_info = ", ".join([f"{i}) {np.sum(labeled_array == i)} pixels" for i in range(1, num_features + 1)])
        print(cluster_info)

        for i in range(1, num_features + 1):
            cluster_size = np.sum(labeled_array == i)
            if cluster_size < min_pixel_threshold:
                pixel_array[labeled_array == i] = 0

        if valid_clusters > 1:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Please draw only one digit at a time to ensure the CNN works accurately. \n Use the clear button to clear the canvas.")
            msg_box.exec()
            return

        self.predict_digit(pixel_array)

    def predict_digit(self, pixel_array):
        if np.sum(pixel_array) == 0:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Please draw a digit before predicting.")
            msg_box.exec()
            return

        cropped_array = self.canvas.cropToBoundingBox(pixel_array)

        cropped_height, cropped_width = cropped_array.shape
        cropped_image = QImage(cropped_width, cropped_height, QImage.Format_RGB32)
        cropped_image.fill(Qt.white)

        for y in range(cropped_height):
            for x in range(cropped_width):
                if cropped_array[y, x] == 1:
                    cropped_image.setPixelColor(x, y, QColor(0, 0, 0))

        cropped_pixmap = QPixmap.fromImage(cropped_image).scaled(
            self.cropped_display.width(),
            self.cropped_display.height(),
            Qt.KeepAspectRatio
        )
        self.cropped_display.setPixmap(cropped_pixmap)

        resized_array = self.canvas.resizeTo28x28(cropped_array)

        resized_image = QImage(28, 28, QImage.Format_RGB32)
        resized_image.fill(Qt.white)

        for y in range(28):
            for x in range(28):
                if resized_array[y, x] == 1:
                    resized_image.setPixelColor(x, y, QColor(0, 0, 0))

        resized_pixmap = QPixmap.fromImage(resized_image).scaled(
            self.resized_display.width(),
            self.resized_display.height(),
            Qt.KeepAspectRatio
        )
        self.resized_display.setPixmap(resized_pixmap)

        digit_array = resized_array.astype('float32').reshape(1, 28, 28, 1)
        prediction = model.predict(digit_array)
        predicted_digit = np.argmax(prediction)

        self.update_bar_graph(prediction[0])

        print("Class probabilities:")
        for i, prob in enumerate(prediction[0]):
            print(f"Class {i}: {prob:.4f}")

        self.predicted_label.setText(f"Prediction: {predicted_digit}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
