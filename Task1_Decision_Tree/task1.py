import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

print("\n===== WINE DATASET =====")

data = load_wine()

X = data.data
y = data.target

feature_names = data.feature_names
target_names = data.target_names

print("Dataset Shape:", X.shape)
print("Number of Features:", len(feature_names))
print("Wine Classes:", target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\n===== GINI VS ENTROPY =====")

gini_model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

gini_model.fit(X_train, y_train)
entropy_model.fit(X_train, y_train)

gini_predictions = gini_model.predict(X_test)
entropy_predictions = entropy_model.predict(X_test)

gini_accuracy = accuracy_score(y_test, gini_predictions)
entropy_accuracy = accuracy_score(y_test, entropy_predictions)

print("Gini Test Accuracy:", round(gini_accuracy, 4))
print("Entropy Test Accuracy:", round(entropy_accuracy, 4))

print("\n===== HYPERPARAMETER TUNING =====")

parameter_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": [1, 2, 3, 4, 5]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    parameter_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:",
      round(grid_search.best_score_, 4))

best_predictions = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_predictions)

print("Best Test Accuracy:", round(best_accuracy, 4))


print("\n===== DECISION TREE VISUALIZATION =====")

plt.figure(figsize=(22, 12))

plot_tree(
    best_model,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree - Wine Dataset (Maximum Depth 3)")
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()

print("Decision Tree image saved as decision_tree.png")
print("\n===== FIRST TWO LEVELS OF TREE SPLITS =====")

tree = best_model.tree_

def explain_split(node, description):
    feature_index = tree.feature[node]

    if feature_index < 0:
        print(description + ": This is a leaf node.")
        return

    feature = feature_names[feature_index]
    threshold = tree.threshold[node]

    print(
        f"{description}: If {feature} <= {threshold:.4f}, "
        "go left; otherwise, go right."
    )

explain_split(0, "First split")

left_child = tree.children_left[0]
right_child = tree.children_right[0]

explain_split(left_child, "Second split - Left branch")
explain_split(right_child, "Second split - Right branch")

print("\n===== TOP 5 IMPORTANT FEATURES =====")

importance_table = pd.DataFrame({
    "Feature": feature_names,
    "Importance": best_model.feature_importances_
})

importance_table = importance_table.sort_values(
    by="Importance",
    ascending=False
)

print(importance_table.head(5).to_string(index=False))

print("\n===== GINI IMPURITY =====")
print("1. Gini impurity measures how mixed the classes are in a tree node.")
print("2. A Gini value of 0 means the node contains only one class.")
print("3. Decision Trees choose splits that reduce Gini impurity.")

