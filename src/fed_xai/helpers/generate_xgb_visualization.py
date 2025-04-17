import xgboost as xgb
from matplotlib import pyplot as plt


def generate_xgb_visualization(bst: xgb.Booster) -> None:
    fig, ax = plt.subplots(figsize=(30, 30))
    xgb.plot_tree(bst, ax=ax, tree_idx=1)
    print(xgb.build_info())
    print(xgb.config.get_config())
    plt.savefig("output/tree.pdf")

    # Create feature importance plot
    fig_importance, ax_importance = plt.subplots(figsize=(10, 10))
    xgb.plot_importance(bst, ax=ax_importance)
    plt.savefig("output/importance.pdf")
