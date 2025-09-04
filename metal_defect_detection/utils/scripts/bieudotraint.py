import os
import matplotlib.pyplot as plt
import numpy as np

def count_images_in_split(data_dir, split):
    """Count images in each class for a given split (train/val/test)."""
    split_dir = os.path.join(data_dir, split)
    class_counts = {}

    for class_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))])

    return class_counts

def plot_dataset_distribution(data_dir):
    """Plot a bar chart showing the distribution of train, val, and test datasets."""
    splits = ['train', 'val', 'test']
    split_counts = {split: count_images_in_split(data_dir, split) for split in splits}

    # Get all class names
    all_classes = sorted(set(class_name for counts in split_counts.values() for class_name in counts.keys()))

    # Prepare data for plotting
    train_counts = [split_counts['train'].get(cls, 0) for cls in all_classes]
    val_counts = [split_counts['val'].get(cls, 0) for cls in all_classes]
    test_counts = [split_counts['test'].get(cls, 0) for cls in all_classes]

    x = np.arange(len(all_classes))
    width = 0.25

    # Plot bars
    plt.bar(x - width, train_counts, width, label='Train', color='blue')
    plt.bar(x, val_counts, width, label='Validation', color='orange')
    plt.bar(x + width, test_counts, width, label='Test', color='green')

    # Add labels and title
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Dataset Distribution: Train, Validation, Test')
    plt.xticks(x, all_classes, rotation=45)
    plt.legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = "metal_defect_detection/data_train"  # Adjust the path to your data directory if needed
    plot_dataset_distribution(data_dir)