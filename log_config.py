

import logging
import os

# Configure logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'main.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Create a console handler so that logs are also displayed on the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)


if __name__ == "__main__":
    import main
    main.main()
