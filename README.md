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
- Modify the CNN architecture and hyperparameters in the `cnn.py` file to improve performance:
  - Number of filters and neurons.
  - Learning rate and activation functions.
  - Batch size, dropout rate, and epochs.

---

## System Requirements

- **Operating System**: Windows, macOS, Linux
- **Python Version**: 3.10+
- **Dependencies**:
  - `tensorflow`
  - `keras`
  - `numpy`
  - `matplotlib`
  - `PySide6`
  - `scipy`
  - `cryptography`

---

## Installation and Usage

This guide provides a step-by-step process for installing and using the Digit Recognition App in PyCharm Community Edition with Python 3.10.

### Prerequisites
Before installing the application, please ensure that you have the following:
1. **PyCharm Community Edition** ([Download Here](https://www.jetbrains.com/pycharm/download/))
2. **Python 3.10** ([Download Here](https://www.python.org/downloads/))

### Installation
1. Download the ZIP file from the provided submission.
2. Extract the ZIP file to a folder on your computer.
3. Open **PyCharm Community Edition** and select **"Open"** from the welcome screen.
4. Navigate to the extracted folder and open the project.
5. Ensure **Python 3.10** is selected as the interpreter:
   - Go to `File > Settings > Project: wgu-c964-digit-recognition > Python Interpreter`.
   - Select **Python 3.10**.
6. Install dependencies automatically:
   - PyCharm will detect a `requirements.txt` file and prompt for installation.
   - If prompted, click **"Install Requirements"** to install the necessary dependencies.
   - To manually install dependencies, navigate to the PyCharm console and run:
     ```bash
     pip install -r requirements.txt
     ```

### Running the Application
1. Open the `main.py` file in **PyCharm**.
2. At the top right of the screen, click on the **Play** button to run the main script.
   - Click `Run > Run 'main'` or press `Shift + F10`.

### Testing the Application
1. Once the application starts, the GUI will launch, allowing you to select or train a new CNN model for digit recognition.
2. Once the model has been selected or trained, the application's main window will appear, allowing the user to draw digits on a canvas.

### Uninstalling the Application
If you no longer need the application:
1. Simply delete the project folder.
2. **Optional**: Remove installed dependencies by running:
   ```bash
   pip uninstall -r requirements.txt -y
   ```

---

## Troubleshooting

- **Invalid Password**: Ensure the correct password is provided when loading encrypted models.
- **Low Prediction Confidence**: Improve the model by tweaking hyperparameters or training duration.

---

## License

This project is developed as part of the WGU Computer Science Capstone Project for educational use. For further inquiries, contact the project author.

**Rafael Estrella Paz**  
📧 Email: [rafael.estrella05@gmail.com](mailto:rafael.estrella05@gmail.com)

