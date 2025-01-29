import sys
import os
from PySide6.QtWidgets import QApplication
import logging

# Import custom modules
import log_config
import state
from cnn import load_mnist_data

def main():
    """Main entry point for the application.

    Initializes the application, sets up the environment, and starts the GUI.
    """
    from view.model_selection_view import ModelSelectionWindow

    try:
        # Load MNIST dataset
        logging.info("Loading MNIST dataset.")
        load_mnist_data()

        # Initialize the application
        app = QApplication(sys.argv)
        logging.info("Application initialized.")

        # Ensure the 'models' directory exists
        if not os.path.exists("models"):
            os.makedirs("models")
            logging.info("Created 'models' directory.")

        # Get a list of models in the 'models' directory
        state.model_list = os.listdir("models")

        #filter out non .keras files
        state.model_list = [model for model in state.model_list if model.endswith(".keras")]

        logging.info(f"Models found: {state.model_list}")

        # Launch the model selection window
        model_selection_window = ModelSelectionWindow()
        logging.info("Model selection window launched.")

        # Start the Qt application event loop
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("-------------------------")
    print("Starting main.py")
    print("-------------------------")
    main()
