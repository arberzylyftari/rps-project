# src/eval.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def plot_history(history, title: str = "Model"):
    """Plot training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Training", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].axhline(y=0.74, color="r", linestyle="--", label="Target (74%)", linewidth=1.5)
    axes[0].set_title(f"{title} - Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Training", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title(f"{title} - Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    max_val_acc = max(history.history["val_accuracy"])
    best_epoch = history.history["val_accuracy"].index(max_val_acc) + 1
    print(f"Best val accuracy: {max_val_acc:.2%} at epoch {best_epoch}")


def plot_confusion_matrix(y_true, y_pred, class_names, title: str = ""):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title or "Confusion Matrix", fontsize=13, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def _pred_labels(model, X):
    """
    Convert model outputs to class labels for both:
    - Keras/TensorFlow models (predict -> probs/logits)
    - sklearn models (predict -> labels, sometimes predict_proba -> probs)
    """
    # 1) Try Keras-style predict(X, verbose=0)
    try:
        y_hat = model.predict(X, verbose=0)
    except TypeError:
        # sklearn predict has no verbose
        y_hat = model.predict(X)

    y_hat = np.asarray(y_hat)

    # (N, C) -> multiclass probs/logits
    if y_hat.ndim == 2:
        if y_hat.shape[1] > 1:
            return np.argmax(y_hat, axis=1)
        # (N, 1) -> binary prob/logit-ish
        return (y_hat.reshape(-1) >= 0.5).astype(int)

    # (N,) already labels
    return y_hat.reshape(-1)


def evaluate_model(
    model,
    X_val,
    y_val,
    X_test,
    y_test,
    class_names,
    show_plots: bool = True,
    print_reports: bool = True,
):
    """
    Full evaluation: accuracy, macro F1, classification report, and confusion matrices
    for both validation and test sets.

    Returns:
        dict with scores so notebook can build comparison table.
        {
          "val": {"accuracy": float, "f1_macro": float},
          "test": {"accuracy": float, "f1_macro": float}
        }
    """
    # Robust predictions (works for Keras + sklearn)
    y_val_pred = _pred_labels(model, X_val)
    y_test_pred = _pred_labels(model, X_test)

    # Metrics
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Macro F1
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

    # Reports as dict so we can print per-class F1 prominently
    val_report = classification_report(
        y_val, y_val_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    test_report = classification_report(
        y_test, y_test_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    if print_reports:
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        print(f"Validation Accuracy: {val_acc:.2%}")
        print(f"Validation F1 Macro: {val_f1_macro:.2%}\n")
        print("Classification report (per-class F1 is the key column):")
        print(classification_report(y_val, y_val_pred, target_names=class_names, zero_division=0))

        print("Per-class F1 (VAL):")
        for cls in class_names:
            if cls in val_report:
                print(f"  - {cls}: {val_report[cls]['f1-score']:.4f}")

        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Test Accuracy: {test_acc:.2%}")
        print(f"Test F1 Macro: {test_f1_macro:.2%}\n")
        print("Classification report (per-class F1 is the key column):")
        print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))

        print("Per-class F1 (TEST):")
        for cls in class_names:
            if cls in test_report:
                print(f"  - {cls}: {test_report[cls]['f1-score']:.4f}")

        print("=" * 70)

    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        cm_val = confusion_matrix(y_val, y_val_pred)
        sns.heatmap(
            cm_val,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
        )
        axes[0].set_title(
            f"Validation (Acc: {val_acc:.2%}, F1: {val_f1_macro:.2%})",
            fontweight="bold",
        )
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")

        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(
            cm_test,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
        )
        axes[1].set_title(
            f"Test (Acc: {test_acc:.2%}, F1: {test_f1_macro:.2%})",
            fontweight="bold",
        )
        axes[1].set_ylabel("True Label")
        axes[1].set_xlabel("Predicted Label")

        plt.tight_layout()
        plt.show()

    # Return scores for notebook
    return {
        "val": {"accuracy": val_acc, "f1_macro": val_f1_macro},
        "test": {"accuracy": test_acc, "f1_macro": test_f1_macro},
    }
