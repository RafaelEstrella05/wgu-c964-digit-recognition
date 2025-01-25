# Digit Recognition Tool (WGU C964 Capstone Project)

This project is an interactive digit recognition application designed for developers and researchers to gain hands-on experience with Convolutional Neural Networks (CNNs) for digit recognition. The application is built using Python and the PySide6 framework for the graphical user interface (GUI), leveraging the MNIST dataset for training and testing.

---

## Features

### Model Training and Testing
- **Train New Models**: Train a CNN model on the MNIST dataset.
- **Test Accuracy**: Evaluate the model's accuracy using the MNIST test dataset.
- **Save and Load Models**: Save trained models with AES-GCM encryption for secure storage and reload them later.

### Interactive GUI
- **Digit Drawing Canvas**: Draw a digit on a canvas and view predictions in real-time.
- **Visualization Tools**:
  - Visualize preprocessing steps (e.g., cropping, resizing).
  - Display prediction confidence using a bar graph.
  - Plot confusion matrices for performance evaluation.
  - Visualize embeddings using scatter plots with dimensionality reduction.

### CNN Customization and Optimization
- Modify the CNN architecture and hyperparameters in the cnn.py file  to improve performance:
  - Number of filters and neurons.
  - Learning rate and activation functions.
  - Batch size, dropout rate, and epochs.

---

## System Requirements

- **Python Version**: 3.10+
- **Dependencies**:
  - `tensorflow`
  - `keras`
  - `numpy`
  - `matplotlib`
  - `PySide6`
  - `scipy`
  - `cryptography`

Install dependencies via pip:
```bash
pip install tensorflow keras numpy matplotlib PySide6 scipy cryptography
```

---

## Installation and Setup

### PyCharm Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RafaelEstrella05/wgu-c964-digit-recognition.git
   cd wgu-c964-digit-recognition
   ```
2. **Open the Project in PyCharm**:
   - Launch PyCharm and select "Open" to navigate to the project folder.
3. **Configure Python Interpreter**:
   - Go to `File > Settings > Project: <project-name> > Python Interpreter`.
   - Set the interpreter to Python 3.10 or later.
4. **Install Dependencies**:
   - Use PyCharm's terminal or the "+" button in the Python Interpreter settings to install required packages.
5. **Run the Application**:
   - Open `main.py` in the editor and click the "Run" button to launch the application.

### Manual Setup (Optional)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RafaelEstrella05/wgu-c964-digit-recognition.git
   
   cd wgu-c964-digit-recognition
   ```
2. **Install Dependencies**:
   ```bash
   pip install tensorflow keras numpy matplotlib PySide6 scipy cryptography
   ```
3. **Run the Application**:
   ```bash
   python main.py
   ```

---

## Usage Instructions

### GUI Workflow
1. **Launch the App**: Open the application to start exploring digit recognition.
2. **Select or Train a Model**:
   - Choose an existing model from the dropdown list.
   - Train a new model by entering a unique model name and password.
3. **Draw and Predict**:
   - Use the canvas to draw a digit.
   - View the prediction and confidence scores.
4. **Visualize Results**:
   - Analyze predictions with confusion matrices.
   - Explore embeddings via scatter plots.

---

## Development Overview

The application includes the following modules:

1. **CNN Architecture (`cnn.py`)**:
   - Implements the convolutional neural network and handles training, testing, and saving/loading models.
2. **Encryption Utilities (`encryption.py`)**:
   - Provides AES-GCM encryption for secure model storage.
3. **GUI Components (`main_view.py`, `model_selection_view.py`)**:
   - Implements the digit drawing canvas, model selection, and visualization tools.
4. **Visualization Tools (`visuals.py`)**:
   - Offers tools for confusion matrix plotting and embedding visualization.

---

## Customization and Improvement

### Tweaking the CNN Model
Adjust the parameters in `cnn.py`:
- **Filters**: Modify the number of filters in convolutional layers.
- **Learning Rate**: Change optimizer settings.
- **Activation Functions**: Experiment with activation functions.
- **Training Duration**: Increase the number of epochs or batch size.

### Adding New Features
To extend functionality, consider:
- Incorporating additional datasets for broader use cases.
- Supporting other machine learning tasks beyond digit recognition.

---

## Troubleshooting

- **Invalid Password**: Ensure the correct password is provided when loading encrypted models.
- **Low Prediction Confidence**: Improve the model by tweaking hyperparameters or training duration.

---

## License

This project is developed as part of the WGU Computer Science Capstone Project for educational use. For further inquiries, contact the project author.

Rafael Estrella Paz

Email: rafael.estrella05@gmail.com


