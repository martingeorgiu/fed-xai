from bellatrex import BellatrexExplain
from bellatrex.utilities import predict_helper
from bellatrex.wrapper_class import pack_trained_ensemble
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from fed_xai.data_loaders.loader import load_data


def bellatrex_explainer(model: RandomForestClassifier) -> None:
    # DATA USED for explanation
    X_train, X_test, y_train, y_test = load_data(0, 1)

    # Pretrained RF model should be packed as a list of dicts with the function below.
    clf_packed = pack_trained_ensemble(model)
    btrex_fitted = BellatrexExplain(
        clf_packed, set_up="auto", p_grid={"n_clusters": [1, 2, 3]}, verbose=3
    )
    btrex_fitted = btrex_fitted.fit(None, None)
    y_train_pred = predict_helper(
        model, X_train
    )  # calls, predict or predict_proba, depending on the underlying model

    tuned_method = btrex_fitted.explain(X_test, 3)

    tuned_method.plot_overview(plot_gui=False, show=True)

    tuned_method.plot_visuals(
        plot_max_depth=5, preds_distr=y_train_pred, conf_level=0.9, tot_digits=4
    )
    tuned_method.create_rules_txt("output/rules.txt")
    plt.show()
