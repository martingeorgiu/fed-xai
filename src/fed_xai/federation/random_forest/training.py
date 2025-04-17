import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from fed_xai.data_loaders.loader import load_data_with_smote
from fed_xai.explainers.bellatrex_explainer import bellatrex_explainer


# This technique was not used eventually
def main() -> None:
    X_train, X_test, y_train, y_test = load_data_with_smote(0, 1)
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=1337,
        max_features="sqrt",
        n_jobs=-1,
    )

    classifier = classifier.fit(X_train, y_train)

    y_pred_test = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {acc:.4f}")

    n_nodes = []
    max_depths = []
    no_trees = 0
    for ind_tree in classifier.estimators_:
        no_trees += 1
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f"Number of trees: {no_trees}")
    print(f"Average number of nodes {int(np.mean(n_nodes))}")
    print(f"Average maximum depth {int(np.mean(max_depths))}")
    print(classification_report(y_test, y_pred_test))

    bellatrex_explainer(classifier)


if __name__ == "__main__":
    main()
