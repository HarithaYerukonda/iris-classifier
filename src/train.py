import os
import joblib
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train a Decision Tree on the Iris dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of test set (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save model and plots (default: outputs)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # Train classifier
    clf = DecisionTreeClassifier(random_state=args.random_state)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" Model trained. Test Accuracy: {acc:.2f}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(args.output_dir, "decision_tree_model.pkl")
    joblib.dump(clf, model_path)
    print(f" Model saved to {model_path}")

    # Save decision tree plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(clf, feature_names=iris.feature_names, class_names=target_names, filled=True)
    fig.tight_layout()
    tree_plot_path = os.path.join(args.output_dir, "decision_tree_plot.png")
    fig.savefig(tree_plot_path)
    print(f" Decision tree plot saved to {tree_plot_path}")

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    fig_cm.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig_cm.savefig(cm_path)
    print(f" Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
