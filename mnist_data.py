import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

"""
    This script preprocesses the MNIST dataset and saves:
    1. Full dataset as a single CSV file
    2. Full set of images
    3. Visualization of 100 samples in a png
"""


def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return (x_train, y_train_cat, y_train), (x_test, y_test_cat, y_test)


def get_balanced_samples(x_data, y_data_int, samples_per_class=10):
    """Get balanced samples with equal class representation"""
    # Create a dictionary to store indices for each class
    class_indices = {i: [] for i in range(10)}
    for idx, label in enumerate(y_data_int):
        class_indices[label].append(idx)

    # Randomly select samples from each class
    selected_indices = []
    for digit in range(10):
        indices = np.random.permutation(class_indices[digit])
        selected_indices.extend(indices[:samples_per_class])

    # Shuffle the final selection to mix classes
    np.random.shuffle(selected_indices)
    return x_data[selected_indices], y_data_int[selected_indices], selected_indices


def save_full_csv(x_train, x_test, y_train_int, y_test_int, csv_path='mnist_dataset/mnist_full.csv'):
    """Save full dataset as a single CSV file"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Flatten and concatenate data
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    x_full = np.concatenate([x_train_flat, x_test_flat])
    y_full = np.concatenate([y_train_int, y_test_int])

    # Combine and save
    full_data = np.hstack([x_full, y_full.reshape(-1, 1)])
    np.savetxt(csv_path, full_data, delimiter=',',
               fmt='%.6f,' * 784 + '%d',
               header=','.join([f'pixel{i}' for i in range(784)] + ['label']),
               comments='')
    print(f"Full dataset saved to {csv_path} ({(full_data.nbytes / 1e6):.1f}MB)")


def save_subset_csv(x_train, y_train_int, csv_path='mnist_dataset/mnist_subset_cleaned.csv', num_samples=100):
    """Save a subset of the dataset as a single CSV file"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Select a subset of the data
    x_subset = x_train[:num_samples]
    y_subset = y_train_int[:num_samples]

    # Flatten and concatenate data
    x_subset_flat = x_subset.reshape(x_subset.shape[0], -1)
    y_subset_flat = y_subset.reshape(-1, 1)

    # Combine and save
    subset_data = np.hstack([x_subset_flat, y_subset_flat])
    np.savetxt(csv_path, subset_data, delimiter=',',
               fmt='%.6f,' * 784 + '%d',
               header=','.join([f'pixel{i}' for i in range(784)] + ['label']),
               comments='')
    print(f"Subset dataset saved to {csv_path} ({(subset_data.nbytes / 1e6):.1f}MB)")




def save_full_images(x_data, y_data_int, sample_dir='mnist_dataset/images'):
    """Save full dataset images with metadata"""
    os.makedirs(sample_dir, exist_ok=True)

    for i, (img, label) in enumerate(zip(x_data, y_data_int)):
        plt.imsave(os.path.join(sample_dir, f'{i:05d}_digit_{label}.png'),
                   img.squeeze(), cmap='gray')
    print(f"Saved {len(x_data)} full dataset images to {sample_dir}")

def create_visualization(x_samples, y_samples_int, save_path='mnist_dataset/samples_overview.png'):
    """Create scrollable visualization of samples"""
    plt.figure(figsize=(10, 40))

    for i in range(100):
        plt.subplot(20, 5, i + 1)
        plt.imshow(x_samples[i].squeeze(), cmap='gray')
        plt.title(f"Digit: {y_samples_int[i]}\nIndex: {i:03d}", fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path} (scrollable PNG)")


def main():
    # Configuration
    SAMPLES_PER_CLASS = 10

    # Load data
    (x_train, _, y_train_int), (x_test, _, y_test_int) = load_and_preprocess_data()

    # Save full dataset CSV
    #save_full_csv(x_train, x_test, y_train_int, y_test_int)

    # Get balanced samples from training set
    x_samples, y_samples_int, original_indices = get_balanced_samples(
        x_train, y_train_int, SAMPLES_PER_CLASS
    )

    # Save full dataset images
    #save_full_images(x_train, y_train_int)

    # Save subset CSV
    #save_subset_csv(x_samples, y_samples_int)

    # Create visualization
    #create_visualization(x_samples, y_samples_int)


if __name__ == "__main__":
    main()