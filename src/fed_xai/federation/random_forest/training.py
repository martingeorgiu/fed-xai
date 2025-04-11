import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from fed_xai.data_loaders import load_data
from fed_xai.explainers.bellatrex_explainer import bellatrex_explainer


def show_confusion_matrix(y_test, y_pred_test, class_names):
    matrix = confusion_matrix(y_test, y_pred_test)
    matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16, 7))
    sns.set_theme(font_scale=1.4)
    sns.heatmap(
        matrix, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Greens, linewidths=0.2
    )
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix for Random Forest Model")
    plt.show()


if __name__ == "__main__":

    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(0, 1)
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=1337,
        max_features="sqrt",
        n_jobs=-1,
        # verbose=1,
    )

    X = train_dmatrix.get_data()

    X_test = valid_dmatrix.get_data()
    y = train_dmatrix.get_label()
    y_test = valid_dmatrix.get_label()
    classifier = classifier.fit(X, y)
    n_nodes = []
    max_depths = []

    y_pred_test = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {acc:.4f}")

    for ind_tree in classifier.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    # print(f"Average number of nodes {int(np.mean(n_nodes))}")
    # print(f"Average maximum depth {int(np.mean(max_depths))}")

    show_confusion_matrix(y_test, y_pred_test, train_dmatrix.feature_names)

    # fn = data.feature_names
    # cn = data.target_names

    # Show trees
    # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
    # for index in range(0, 5):
    #     tree.plot_tree(
    #         classifier.estimators_[index],
    #         # feature_names=fn,
    #         # class_names=cn,
    #         filled=True,
    #         ax=axes[index],
    #     )

    # axes[index].set_title("Estimator: " + str(index), fontsize=11)
    # fig.savefig("rf_5trees.png")
    # bellatrex_explainer(classifier)
