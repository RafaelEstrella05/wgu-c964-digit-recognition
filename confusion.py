import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QListWidget, QLabel, QWidget, QMessageBox, QApplication
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

global model_list
global model_name
global model
global x_test
global y_test

def plot_confusion_matrix(confusion_matrix, classes):
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
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

def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[true_label, predicted_label] += 1

    return confusion_matrix

'''
This window is in charge of reading the contents of the /models directory and displaying the available models in a list 
for the user to select. The models are displayed in a dropdown list, in which the default option is to train a new model.
the name of the new model will be "mnist_model_v_X.keras" where X is the next available number in the sequence.
The "continue" button allows the user to continue with or without selecting a model.
'''
class ModelSelectionWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Model Selection")
        self.setGeometry(100, 100, 800, 600)

        self.model_list_label = QLabel("Select a model to use:")
        self.model_list_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.model_list_widget = QListWidget()
        self.model_list_widget.setStyleSheet("font-size: 12px;")
        self.model_list_widget.addItem("+ Train a new model")

        for m in model_list:
            self.model_list_widget.addItem(m)

        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_button_clicked)
        self.continue_button.setStyleSheet("font-size: 14px; padding: 10px;")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.model_list_label)
        self.layout.addWidget(self.model_list_widget)
        self.layout.addWidget(self.continue_button)

        self.setLayout(self.layout)
        self.show()
        self.model_selection_window = self

    def continue_button_clicked(self):
        # Get the selected model
        selected_model_name = self.model_list_widget.currentItem().text()
        print(selected_model_name)

        #verify that the extension is .keras
        if selected_model_name[-6:] != ".keras":
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Please select a keras model to continue.")
            msg_box.exec()
            return

        load_or_train_model(selected_model_name)

        self.model_selection_window.close()
        main()

def load_or_train_model(model_file):
    global model
    global model_name
    global x_test
    global y_test

    # Load MNIST dataset and preprocess
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    #y_test = to_categorical(y_test, 10)
    y_test = y_test

    if model_file == "+ Train a new model":
        model_file = f"models/mnist_model_v_{len(model_list) + 1}.keras"

        #extract the model name from the file path
        model_name = model_file.split("/")[1]

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
    else:
        file_name = f"models/{model_file}"

        # load the model from the models directory
        model = load_model(file_name)

        # remove the models/ prefix from the model name
        model_name = model_file.split("/")[0]

def main():
    global model
    global model_name
    global x_test
    global y_test

    # Predict labels
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)

    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(y_test, y_pred, num_classes=10)

    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix, classes=[str(i) for i in range(10)])

if __name__ == "__main__":
    global model_list

    app = QApplication(sys.argv)

    # get list of models from /models directory
    model_list = os.listdir("models")

    model_selection_window = ModelSelectionWindow()


    sys.exit(app.exec())
