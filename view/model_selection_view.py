from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, \
    QMessageBox, QListWidget, QLineEdit, QInputDialog
import logging

import log_config
import state
from cnn import load_cnn_model, train_new_model, evaluate_model_accuracy
from view.main_view import MainWindow

'''

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

        for m in state.model_list:
            self.model_list_widget.addItem(m)

        self.model_list_widget.currentItemChanged.connect(lambda: self.password_field.setPlaceholderText("Enter Decryption Password" if self.model_list_widget.currentItem().text()[-6:] == ".keras" else "Enter Encryption Password"))

        #add password field so that the user can enter the password to decrypt the model
        self.password_field = QLineEdit()
        self.password_field.setPlaceholderText("Enter Encryption Password")
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
        pw = self.password_field.text()

        # if password is not entered, display an error message
        if pw == "":
            logging.warning("No password entered")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)

            # if + Train a new model is selected, display a different message
            if self.model_list_widget.currentItem().text() == "+ Train a new model":
                msg_box.setText(f"Please enter a password to encrypt the new model.")
            else:
                msg_box.setText(f"Please enter the password to decrypt the model.")

            msg_box.exec()
            return

        selection = self.model_list_widget.currentItem().text()

        #if selection is to train a new model
        if selection == "+ Train a new model":

            #confirm the password with a dialog box "Confirm Password"
            confirm_pw, ok = QInputDialog.getText(self, "Confirm Password", "Please confirm the password to encrypt the new model:", QLineEdit.Password)
            if not ok:
                return
            if confirm_pw != pw:
                logging.warning("Passwords do not match")
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setText(f"Passwords do not match. Please try again.")
                msg_box.exec()
                return


            train_new_model(pw)


        else:
            # Get the selected model
            selected_model_name = self.model_list_widget.currentItem().text()
            logging.info(f"Selected Option: {selected_model_name}")

            #if selected model does not have .keras extension, display an error message
            if selected_model_name[-6:] != ".keras":
                logging.warning("Invalid model file selected")
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setText(f"Please select a keras model to continue.")
                msg_box.exec()
                return

            load_cnn_model(selected_model_name, pw)

            if state.model is None:
                return

        evaluate_model_accuracy()

        self.model_selection_window.close()
        main_window = MainWindow()
        main_window.show()

if __name__ == "__main__":
    print("Please run main.py to start the application.")