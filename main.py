import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PySide6.QtGui import QPainter, QMouseEvent, QImage, QColor, QPixmap, QPen
from PySide6.QtCore import Qt, QPoint
from scipy.ndimage import label

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

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.predict_button)

        grid_layout = QHBoxLayout()

        left_grid_layout = QVBoxLayout()
        # Add title of the cropped image
        title_cropped = QLabel("1) Crop, Center and Pad")
        title_cropped.setStyleSheet("font-size: 11px;")
        left_grid_layout.addWidget(title_cropped)

        # Add the cropped image
        left_grid_layout.addWidget(self.cropped_display)

        right_grid_layout = QVBoxLayout()
        # Add title of the resized image
        title_resized = QLabel("2) Resize to 28x28")
        title_resized.setStyleSheet("font-size: 11px;")
        right_grid_layout.addWidget(title_resized)

        # Add the resized image
        right_grid_layout.addWidget(self.resized_display)

        grid_layout.addLayout(left_grid_layout)
        grid_layout.addLayout(right_grid_layout)

        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(self.title_label)
        vbox_layout.addWidget(self.preprocess_label)
        vbox_layout.addLayout(grid_layout)
        vbox_layout.addWidget(self.predicted_label)
        vbox_layout.addLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(vbox_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def validate_and_predict(self):
        pixel_array = self.canvas.exportToArray()

        # Validate the input for multiple clumps
        labeled_array, num_features = label(pixel_array)

        if num_features > 1:
            # Display a message to the user
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Please draw only one digit at a time to ensure the CNN works accurately. \n use the clear button to clear the canvas")
            msg_box.exec()
            return

        # Proceed with prediction if valid
        self.predict_digit(pixel_array)

    def predict_digit(self, pixel_array):
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
        self.predicted_label.setText(f"Prediction: {predicted_digit}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
