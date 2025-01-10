import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from PySide6.QtGui import QPainter, QMouseEvent, QImage, QColor, QPixmap, QPen
from PySide6.QtCore import Qt, QPoint

import os  # Provides functions to interact with the operating system
import tensorflow as tf  # TensorFlow library for building and training machine learning models
from tensorflow.keras.models import load_model  # Function to load a pre-trained Keras model
from tensorflow.keras.datasets import mnist  # MNIST dataset, a benchmark for digit recognition tasks
from tensorflow.keras.utils import to_categorical  # Utility to convert labels to one-hot encoding

# File name for saving and loading the trained model
model_file = "mnist_model.keras"

# Load the MNIST dataset and split it into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data() # x_train is the training images, y_train is the labels

# Normalize the pixel values of the images to the range [0, 1] and reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to one-hot encoding for compatibility with the softmax output layer
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
'''
EXAMPLE: 
Before one-hot encoding:
y_train = [2, 4, 2, 1, 3, 4, 6, 7, 8, 5, 3, ...]

After one-hot encoding:
y_train =
[
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Label 2
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Label 4
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Label 2
 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Label 1
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Label 3
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Label 4
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Label 6
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Label 7
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Label 8
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Label 5
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Label 3
 ...
]
'''
# Check if a saved model exists
if os.path.exists(model_file):
    # Load the saved model if it exists
    model = load_model(model_file)
    print("Model loaded successfully.")
else:
    # Define a new Convolutional Neural Network (CNN) architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # First convolutional layer
        tf.keras.layers.MaxPooling2D((2, 2)),  # First max pooling layer to reduce spatial dimensions
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
        tf.keras.layers.MaxPooling2D((2, 2)),  # Second max pooling layer
        tf.keras.layers.Flatten(),  # Flatten the output to feed into the dense layers
        tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons for digit classification
    ])

    # Compile the model with Adam optimizer and categorical crossentropy loss function
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model on the training data with 5 epochs and a batch size of 128
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

    # Save the trained model to the file
    model.save(model_file)
    print("Model trained and saved.")

# Evaluate the model's accuracy on the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")


class Canvas(QWidget):
    def __init__(self, main_window, width=400, height=400):
        super().__init__()
        self.main_window = main_window
        self.setFixedSize(width, height)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()
        self.pen_thickness = 9  # Set brush size here

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and event.buttons() == Qt.LeftButton:
            painter = QPainter(self.image)
            pen = QPen(QColor(0, 0, 0))
            pen.setWidth(self.pen_thickness)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(0, 0, self.image)

    def clearCanvas(self):
        self.image.fill(Qt.white)
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

        # Calculate proportional padding
        height, width = pixel_array.shape
        box_height = y_max - y_min + 1
        box_width = x_max - x_min + 1
        padding = max(2, int(0.15 * min(box_height, box_width)))

        # Determine the size of the square bounding box
        box_size = max(box_height, box_width) + 2 * padding

        # Add padding and make the box square
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        y_min = max(0, center_y - box_size // 2)
        y_max = min(height - 1, center_y + box_size // 2)
        x_min = max(0, center_x - box_size // 2)
        x_max = min(width - 1, center_x + box_size // 2)

        cropped_array = pixel_array[y_min:y_max + 1, x_min:x_max + 1]
        return cropped_array

    def resizeTo28x28(self, cropped_array):
        target_size = (28, 28)
        resized_array = np.zeros(target_size, dtype=np.uint8)

        cropped_height, cropped_width = cropped_array.shape
        for y in range(target_size[0]):
            for x in range(target_size[1]):
                # Map the 28x28 grid to the size of the cropped array
                src_y = int(y * cropped_height / target_size[0])
                src_x = int(x * cropped_width / target_size[1])
                resized_array[y, x] = cropped_array[src_y, src_x]

        return resized_array

    def predict_digit(self, digit_array): #digit_array is numpy array
        prediction = model.predict(digit_array)  # Use the CNN model to predict the digit
        predicted_digit = np.argmax(prediction)  # Extract the digit with the highest probability
        self.main_window.predicted_label.setText(f"Predicted Digit: {predicted_digit}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("WGU C964 - MNIST Digit Recognizer")
        self.canvas = Canvas(self)

        self.clear_button = QPushButton("Clear Canvas")
        self.clear_button.clicked.connect(self.canvas.clearCanvas)

        self.export_button = QPushButton("Predict Digit")
        self.export_button.clicked.connect(self.exportCropAndResizeCanvas)

        # Display the cropped and resized images
        self.cropped_display = QLabel()
        self.cropped_display.setFixedSize(200, 200)
        self.cropped_display.setStyleSheet("border: 1px solid black;")

        self.resized_display = QLabel()
        self.resized_display.setFixedSize(200, 200)
        self.resized_display.setStyleSheet("border: 1px solid black;")

        self.predicted_label = QLabel("Predicted Digit: None")
        self.predicted_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.export_button)
        layout.addWidget(self.predicted_label)

        display_layout = QHBoxLayout()
        display_layout.addLayout(layout)
        display_layout.addWidget(self.cropped_display)
        display_layout.addWidget(self.resized_display)

        container = QWidget()
        container.setLayout(display_layout)
        self.setCentralWidget(container)

    def exportCropAndResizeCanvas(self):
        pixel_array = self.canvas.exportToArray()
        cropped_array = self.canvas.cropToBoundingBox(pixel_array)

        # Convert the cropped array to a QImage for display
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

        # Resize the cropped array to 28x28
        resized_array = self.canvas.resizeTo28x28(cropped_array)

        # Convert the resized array to a QImage for display
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

        # Save resized array for prediction
        self.resized_array = resized_array

        # Predict the digit using the resized array, convert into numpy array
        digit_array = resized_array.astype('float32').reshape(1, 28, 28, 1)
        self.canvas.predict_digit(digit_array)

        #click the clear button
        self.canvas.clearCanvas()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
