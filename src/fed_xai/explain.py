import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score  # noqa: F401
from sklearn.metrics import roc_auc_score

from fed_xai.data_loaders.loader import load_data_for_xgb
from fed_xai.explainers.combining_rulecosi_explainer import combining_rulecosi_explainer
from fed_xai.helpers.accuracy_score_with_threshold import accuracy_score_with_threshold
from fed_xai.helpers.number_of_trees import get_number_of_trees
from fed_xai.xgboost.booster_to_classifier import booster_to_classifier
from fed_xai.xgboost.const import booster_params_from_hp


def generate_viz(bst: xgb.Booster) -> None:
    fig, ax = plt.subplots(figsize=(30, 30))
    xgb.plot_tree(bst, ax=ax, tree_idx=1)
    print(xgb.build_info())
    print(xgb.config.get_config())
    # Save tree visualization
    plt.savefig("output/tree.pdf")

    # Create feature importance plot
    fig_importance, ax_importance = plt.subplots(figsize=(10, 10))
    xgb.plot_importance(bst, ax=ax_importance)
    plt.savefig("output/importance.pdf")
    bst.dump_model("output/dump.json")


def get_stats(bst: xgb.Booster) -> None:
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data_for_xgb(0, 1)

    y_pred = bst.predict(valid_dmatrix, validate_features=False)
    y_true = valid_dmatrix.get_label()

    num_trees = len(bst.get_dump())
    print(f"Number of trees: {num_trees} trees")
    print("----accuracy----")
    print(accuracy_score_with_threshold(y_true, y_pred))
    print("----roc_auc_score----")
    print(roc_auc_score(y_true, y_pred))

    eval_results = bst.eval_set(
        evals=[(valid_dmatrix, "valid")],
        iteration=bst.num_boosted_rounds() - 1,
    )
    print("----eval_results----")
    print(eval_results)


def main() -> None:
    with open("output/output2.bin", "rb") as file:
        data = file.read()
    bst = xgb.Booster(params=booster_params_from_hp)
    para_b = bytearray(data)
    bst.load_model(para_b)

    clf = booster_to_classifier(bst)
    no_trees = get_number_of_trees(bst)
    print(f"no_trees: {no_trees}")
    combining_rulecosi_explainer(clf)
    get_stats(bst)
    # generate_viz(bst)
    # generate_rules(bst)
    # shap_explainer(bst)
    # bellatrex_explainer(bst)


if __name__ == "__main__":
    main()
