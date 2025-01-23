# Global state module for storing application-wide variables

# Model-related variables
model = None  # The currently loaded TensorFlow model
model_name = None  # The name of the currently loaded model
model_list = []  # List of available models in the 'models' directory
model_accuracy = None  # Accuracy of the currently loaded model

# Data-related variables
x_train_data = None  # Training data (features)
y_train_data = None  # Training data (labels)
x_test_data = None  # Test data (features)
y_test_data = None  # Test data (labels)

if __name__ == "__main__":
    print("This module defines the application's global state.")
