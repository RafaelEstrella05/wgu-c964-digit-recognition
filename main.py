import sys
import os
from PySide6.QtWidgets import QApplication
import logging

#import custom modules
import log_config
import state
from cnn import load_mnist_data

if __name__ == "__main__":
    from view.model_selection_view import ModelSelectionWindow

    load_mnist_data()

    try:
        app = QApplication(sys.argv)
        logging.info("Starting application")

        # get list of models from /models directory
        state.model_list = os.listdir("models")

        # log the models found in the directory
        logging.info(f"Models found: {state.model_list}")

        model_selection_window = ModelSelectionWindow()

        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit()
