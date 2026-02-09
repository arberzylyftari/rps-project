# src/eval.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_history(history, title: str = "Model"):
    """Plot training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].axhline(y=0.74, color='r', linestyle='--', label='Target (74%)', linewidth=1.5)
    axes[0].set_title(f'{title} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    max_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(max_val_acc) + 1
    print(f"Best val accuracy: {max_val_acc:.2%} at epoch {best_epoch}")


def plot_confusion_matrix(y_true, y_pred, class_names, title: str = ""):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(title or 'Confusion Matrix', fontsize=13, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_val, y_val, X_test, y_test, class_names):
    """
    Full evaluation: accuracy, classification report, and confusion matrices
    for both validation and test sets.
    """
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"Validation Accuracy: {val_acc:.2%}\n")
    print(classification_report(y_val, y_val_pred, target_names=class_names))

    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Test Accuracy: {test_acc:.2%}\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_val = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'Validation Set (Acc: {val_acc:.2%})', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'Test Set (Acc: {test_acc:.2%})', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()
