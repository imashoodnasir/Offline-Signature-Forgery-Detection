# visualize_results.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Utility: Plot metric comparison from a results dictionary
def plot_metric_comparison(results_dict, metric_name="Accuracy", save_path=None):
    models = list(results_dict.keys())
    values = [results_dict[m] for m in models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison Across Models")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example dictionary (you should extract this from result logs)
example_accuracy = {
    "MSLG-Net": 0.98,
    "VGG16": 0.92,
    "ResNet50": 0.91,
    "MobileNetV2": 0.88,
    "DenseNet121": 0.90,
    "Xception": 0.89
}

# Plot example
plot_metric_comparison(example_accuracy, "Accuracy", "model_accuracy_comparison.png")

# Confusion matrix plotting
def plot_confusion_matrix(y_true, y_pred, labels=[0, 1], title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
