from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QMessageBox, QListWidget, QLineEdit, QInputDialog
)
import logging

import log_config  # Custom logging configuration
import state  # Global application state
from cnn import load_cnn_model, train_new_model, evaluate_model_accuracy
from view.main_view import MainWindow

class ModelSelectionWindow(QWidget):
    """Window for selecting or training CNN models."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Model Selection")
        self.setGeometry(100, 100, 400, 400)

        # UI Components
        self.model_list_label = QLabel("Select a saved model or train a new one:")
        self.model_list_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.model_list_widget = QListWidget()
        self.model_list_widget.setStyleSheet("font-size: 14px; padding: 2px;")
        self.model_list_widget.addItem("+ Train a new model")

        # Populate the model list
        for model_name in state.model_list:
            self.model_list_widget.addItem(model_name)

        # Password field
        self.password_field = QLineEdit()
        self.password_field.setPlaceholderText("Enter Encryption/Decryption Password")
        self.password_field.setEchoMode(QLineEdit.Password)

        # Continue button
        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_button_clicked)
        self.continue_button.setStyleSheet("font-size: 14px; padding: 10px;")

        # Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.model_list_label)
        self.layout.addWidget(self.model_list_widget)
        self.layout.addWidget(self.password_field)
        self.layout.addWidget(self.continue_button)

        self.setLayout(self.layout)
        self.show()
        logging.info("Model Selection Window initialized.")

    def continue_button_clicked(self):
        """Handle the continue button click event."""
        password = self.password_field.text()
        if not password:
            self.display_warning("Please enter a password.")
            return

        selected_item = self.model_list_widget.currentItem().text()
        if selected_item == "+ Train a new model":
            self.handle_new_model_training(password)
        else:
            self.handle_model_loading(selected_item, password)

    def handle_new_model_training(self, password):
        """Handle training a new model with a password."""
        confirm_password, ok = QInputDialog.getText(
            self, "Confirm Password", "Confirm the password to encrypt the new model:", QLineEdit.Password
        )
        if not ok or confirm_password != password:
            self.display_warning("Passwords do not match. Please try again.")
            return

        train_new_model(password)
        evaluate_model_accuracy()
        self.close_and_open_main_window()

    def handle_model_loading(self, model_name, password):
        """Handle loading an existing model with decryption."""
        if not model_name.endswith(".keras"):
            self.display_warning("Please select a valid Keras model file.")
            return

        load_cnn_model(model_name, password)
        if state.model is None:
            return

        evaluate_model_accuracy()
        self.close_and_open_main_window()

    def close_and_open_main_window(self):
        """Close the current window and open the main application window."""
        self.close()
        main_window = MainWindow()
        main_window.show()

    def display_warning(self, message):
        """Display a warning message box."""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.exec()

if __name__ == "__main__":
    print("Please run main.py to start the application.")
