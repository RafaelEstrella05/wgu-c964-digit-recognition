from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QMessageBox, QListWidget, QLineEdit, QInputDialog, QDialog, QFormLayout, QDialogButtonBox
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

        # Populate the model list, add only .keras files
        for model_name in state.model_list:
            if model_name.endswith(".keras"):
                self.model_list_widget.addItem(model_name)


        # Continue button
        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_button_clicked)
        self.continue_button.setStyleSheet("font-size: 14px; padding: 10px;")

        # Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.model_list_label)
        self.layout.addWidget(self.model_list_widget)
        self.layout.addWidget(self.continue_button)

        self.setLayout(self.layout)
        self.show()
        logging.info("Model Selection Window initialized.")

    def continue_button_clicked(self):
        """Handle the continue button click event."""

        selected_item = self.model_list_widget.currentItem().text()
        if selected_item == "+ Train a new model":
            dialog = ModelTrainingForm()
            if dialog.exec() == QDialog.Accepted:
                model_name, password, confirm_password = dialog.get_inputs()

                self.handle_new_model_training(model_name, password)
        else:
            #make sure the file is a valid model file .keras
            if not selected_item.endswith(".keras"):
                self.display_warning("Please select a valid Keras model file (.keras).")
                return


            password, ok = QInputDialog.getText(self, "Enter Password", "Enter Decryption Password:",
                                                QLineEdit.Password)
            if not ok:
                return

            if not password:
                self.display_warning("Please enter a password.")
                return

            self.handle_model_loading(selected_item, password)

    def handle_new_model_training(self, model_name,  password):
        """Handle training a new model with a password."""


        # display "A New Model Will be Trained: Please watch the console for progress." and wait for confirmation to click ok
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("A New Model Will be Trained: Please watch the console for progress.")

        if msg_box.exec() == QMessageBox.Ok:
            # close the current window
            self.close()

            train_new_model(model_name, password)
            evaluate_model_accuracy()

            #open the main window
            main_window = MainWindow()
            main_window.show()


    def handle_model_loading(self, model_name, password):
        """Handle loading an existing model with decryption."""
        if not model_name.endswith(".keras"):
            self.display_warning("Please select a valid Keras model file.")
            return

        load_cnn_model(model_name, password)
        if state.model is None:
            return

        evaluate_model_accuracy()

        #close the current window
        self.close()
        #open the main window
        main_window = MainWindow()
        main_window.show()

    def display_warning(self, message):
        """Display a warning message box."""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.exec()

class ModelTrainingForm(QDialog):
    """
    Dialog for inputting model name and password for training a new model.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Train New Model")
        self.setGeometry(100, 100, 300, 200)

        self.layout = QFormLayout()

        self.model_name_input = QLineEdit()
        self.model_name_input.textChanged.connect(self.input_changed)
        self.layout.addRow("Model Name:", self.model_name_input)


        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.textChanged.connect(self.input_changed)
        self.layout.addRow("Encryption Password:", self.password_input)


        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input.textChanged.connect(self.input_changed)
        self.layout.addRow("Confirm Password:", self.confirm_password_input)

        #create label for showing message
        self.message_label = QLabel("...")
        self.layout.addRow(self.message_label)

        self.setLayout(self.layout)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        #disable the ok button by default
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

        self.layout.addWidget(self.buttons)

    def input_changed(self):
        """Enable the OK button if all inputs are valid."""
        model_name, password, confirm_password = self.get_inputs()
        invalid_chars = set(r'\/:*?"<>|')
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        if model_name + ".keras" in state.model_list:
            self.message_label.setText("Model name already exists.")
        elif any(char in invalid_chars for char in model_name):
            self.message_label.setText("Model name contains invalid characters.")
        elif len(password) < 8:
            self.message_label.setText("Password is too short.")
        elif password != confirm_password:
            self.message_label.setText("Passwords do not match.")
        else:
            self.message_label.setText("Please fill in all fields.")

    def get_inputs(self):
        return self.model_name_input.text(), self.password_input.text(), self.confirm_password_input.text()

if __name__ == "__main__":
    print("Please run main.py to start the application.")
