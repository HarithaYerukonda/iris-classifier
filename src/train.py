import os
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


def main():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Test Accuracy: {acc:.2f}")

    # Save outputs directory
    os.makedirs("outputs", exist_ok=True)

    # Save model
    model_path = os.path.join("outputs", "decision_tree_model.pkl")
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to {model_path}")

    # Save plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(clf, feature_names=iris.feature_names, class_names=target_names, filled=True)
    fig.tight_layout()
    fig.savefig(os.path.join("outputs", "decision_tree_plot.png"))
    print("🌳 Decision tree plot saved to outputs/decision_tree_plot.png")


if __name__ == "__main__":
    main()