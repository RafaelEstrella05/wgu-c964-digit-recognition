import logging
import os

def configure_logging():
    """Configure logging for the application.

    Sets up logging to a file and the console with appropriate formatting.
    """
    log_file_path = os.path.join(os.path.dirname(__file__), 'main.log')

    # Configure logging to file
    logging.basicConfig(
        filename=log_file_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # Configure logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)

    logging.info("Logging configuration complete. Logs will be written to both the console and the file.")

# Run logging configuration when the module is imported
configure_logging()

if __name__ == "__main__":
    print("This module configures logging for the application.")
