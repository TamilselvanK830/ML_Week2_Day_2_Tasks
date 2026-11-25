import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

print("\n===== WINE DATASET =====")

data = load_wine()

X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset Shape:", X.shape)
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\n===== SINGLE DECISION TREE =====")

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

print("Single Tree Test Accuracy:", round(tree_accuracy, 4))

print("\n===== RANDOM FOREST =====")

rf_model = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("Random Forest Test Accuracy:", round(rf_accuracy, 4))
print("OOB Score:", round(rf_model.oob_score_, 4))

print("\n===== MODEL ACCURACY COMPARISON =====")

print(f"Single Tree Test Accuracy: {tree_accuracy:.4f}")
print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")
print(f"OOB Score: {rf_model.oob_score_:.4f}")

print("\n===== N_ESTIMATORS COMPARISON =====")

estimators = [10, 50, 100, 300]
accuracies = []

for n in estimators:

    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    accuracies.append(accuracy)

    print(f"Trees: {n} | Test Accuracy: {accuracy:.4f}")

plt.figure(figsize=(8, 5))

plt.plot(
    estimators,
    accuracies,
    marker="o",
    linewidth=2
)

plt.title("Random Forest - Number of Trees vs Accuracy")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Test Accuracy")
plt.xticks(estimators)
plt.grid(True)

plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Accuracy graph saved as accuracy_comparison.png")

print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")

importance_table = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance_table = importance_table.sort_values(
    by="Importance",
    ascending=False
)

print(importance_table.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))

plt.barh(
    importance_table["Feature"],
    importance_table["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

print("Feature importance graph saved as feature_importance.png")

print("\n===== BONUS: EXTRA TREES CLASSIFIER =====")

extra_model = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)

extra_model.fit(X_train, y_train)

extra_predictions = extra_model.predict(X_test)

extra_accuracy = accuracy_score(
    y_test,
    extra_predictions
)

print("Extra Trees Test Accuracy:", round(extra_accuracy, 4))

# Compare all models
comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "Extra Trees"
    ],
    "Test Accuracy": [
        tree_accuracy,
        rf_accuracy,
        extra_accuracy
    ]
})

print("\n===== FINAL MODEL COMPARISON =====")
print(comparison.to_string(index=False))

print("\n===== BAGGING EXPLANATION =====")

print("1. Bagging means Bootstrap Aggregation.")
print("2. It trains multiple decision trees using different bootstrap samples.")
print("3. Random Forest combines their predictions using majority voting.")

print("\n===== WHY RANDOM FOREST OVERFITS LESS =====")

print("1. A single decision tree can learn noise from training data.")
print("2. Random Forest combines predictions from many different trees.")
print("3. Combining trees reduces variance and usually improves generalisation.")

