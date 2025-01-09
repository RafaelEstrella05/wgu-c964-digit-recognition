import os  # Provides functions to interact with the operating system
import tensorflow as tf  # TensorFlow library for building and training machine learning models
from tensorflow.keras.models import load_model  # Function to load a pre-trained Keras model
from tensorflow.keras.datasets import mnist  # MNIST dataset, a benchmark for digit recognition tasks
from tensorflow.keras.utils import to_categorical  # Utility to convert labels to one-hot encoding
import numpy as np  # Library for numerical computations
from PySide6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QLabel  # Qt Widgets for GUI
from PySide6.QtGui import QPainter, QPen  # Qt GUI components for painting and drawing
from PySide6.QtCore import Qt  # Qt core functionalities
import sys  # Provides access to system-specific parameters and functions

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

# Define a custom widget to create a pixel grid for digit drawing
class PixelGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_size = 28  # Number of rows and columns in the grid
        self.pixel_size = 20  # Size of each pixel in the grid (in pixels)
        self.pixels = np.zeros((self.grid_size, self.grid_size), dtype=int)  # Initialize grid with zeros
        self.setFixedSize(self.grid_size * self.pixel_size, self.grid_size * self.pixel_size)  # Set widget size

    def paintEvent(self, event):
        painter = QPainter(self)  # Create a painter object for the widget
        pen = QPen(Qt.black)  # Set pen color to black
        pen.setWidth(1)  # Set pen width
        painter.setPen(pen)

        # Loop through each cell in the grid and draw pixels
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect_x = x * self.pixel_size  # X-coordinate of the top-left corner of the cell
                rect_y = y * self.pixel_size  # Y-coordinate of the top-left corner of the cell
                if self.pixels[y, x] == 1:  # Check if the pixel is activated (black)
                    painter.fillRect(rect_x, rect_y, self.pixel_size, self.pixel_size, Qt.black)  # Fill cell with black
                painter.drawRect(rect_x, rect_y, self.pixel_size, self.pixel_size)  # Draw cell border

    def mousePressEvent(self, event):
        self.toggle_pixel(event)  # Toggle pixel on mouse press

    def mouseMoveEvent(self, event):
        self.toggle_pixel(event)  # Toggle pixel on mouse move

    def toggle_pixel(self, event):
        pos = event.position()  # Get the position of the mouse event
        x = int(pos.x() // self.pixel_size)  # Calculate grid column
        y = int(pos.y() // self.pixel_size)  # Calculate grid row
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:  # Check if coordinates are within grid bounds
            self.pixels[y, x] = 1  # Activate the pixel (set to 1)
            self.update()  # Redraw the widget

    def reset_grid(self):
        self.pixels.fill(0)  # Reset all pixels to 0 (clear the grid)
        self.update()  # Redraw the widget

    def get_digit_array(self):
        return self.pixels.astype('float32').reshape(1, 28, 28, 1)  # Reshape grid to match CNN input format

# Define the main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digit Recognizer")  # Set window title

        self.central_widget = QWidget()  # Create central widget
        self.layout = QGridLayout()  # Use a grid layout
        self.central_widget.setLayout(self.layout)  # Set the layout for the central widget
        self.setCentralWidget(self.central_widget)  # Set the central widget for the window

        self.pixel_grid = PixelGrid()  # Create an instance of the PixelGrid widget
        self.layout.addWidget(self.pixel_grid, 0, 0, 1, 2)  # Add the pixel grid to the layout

        self.predict_button = QPushButton("Predict")  # Create a button for predictions
        self.predict_button.clicked.connect(self.predict_digit)  # Connect button to prediction function
        self.layout.addWidget(self.predict_button, 1, 0)  # Add button to the layout

        self.reset_button = QPushButton("Reset")  # Create a button to reset the grid
        self.reset_button.clicked.connect(self.pixel_grid.reset_grid)  # Connect button to grid reset function
        self.layout.addWidget(self.reset_button, 1, 1)  # Add button to the layout

        self.result_label = QLabel("Prediction: ")  # Create a label to display predictions
        self.layout.addWidget(self.result_label, 2, 0, 1, 2)  # Add label to the layout

    def predict_digit(self):
        digit_array = self.pixel_grid.get_digit_array()  # Get the user-drawn digit as a numpy array
        prediction = model.predict(digit_array)  # Use the CNN model to predict the digit
        predicted_digit = np.argmax(prediction)  # Extract the digit with the highest probability
        self.result_label.setText(f"Prediction: {predicted_digit}")  # Display the predicted digit

# Entry point of the application
if __name__ == "__main__":
    app = QApplication(sys.argv)  # Create the application object

    window = MainWindow()  # Create the main application window
    window.show()  # Show the main window

    sys.exit(app.exec())  # Execute the application event loop
