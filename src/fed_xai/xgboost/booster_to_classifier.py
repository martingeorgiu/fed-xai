import json

import xgboost as xgb
from sklearn.metrics import accuracy_score  # noqa: F401

from fed_xai.xgboost.const import booster_params_from_hp


def booster_to_classifier(bst: xgb.Booster) -> xgb.XGBClassifier:
    bst.set_attr(scikit_learn='{"_estimator_type": "classifier"}')
    xgb_classifier = xgb.XGBClassifier(params=booster_params_from_hp)
    load_model(xgb_classifier, bst)
    return xgb_classifier


def load_model(self: xgb.XGBClassifier, bst: xgb.Booster) -> None:
    print("== Loading model ==")
    self._Booster = bst

    meta_str = self.get_booster().attr("scikit_learn")
    if meta_str is not None:
        meta = json.loads(meta_str)
        t = meta.get("_estimator_type", None)
        if t is not None and t != self._get_type():
            raise TypeError(
                f"Loading an estimator with different type. Expecting: {self._get_type()}, got: {t}"
            )

    self.feature_types = self.get_booster().feature_types
    self.get_booster().set_attr(scikit_learn=None)

    # The save_config dumps as strings the numbers :(((
    config = json.loads(self.get_booster().save_config(), object_hook=convert_strings_to_numbers)
    self._load_model_attributes(config)


def try_number(s):  # noqa: ANN001, ANN201
    """Try converting s to an int; if that fails, try float; otherwise return s unchanged."""
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def convert_strings_to_numbers(d):  # noqa: ANN001, ANN201
    for key, value in d.items():
        if isinstance(value, str):
            d[key] = try_number(value)
        elif isinstance(value, dict):
            d[key] = convert_strings_to_numbers(value)
        elif isinstance(value, list):
            # Process each item in the list; if the item is a dict, process it recursively,
            # if it's a string, attempt a conversion.
            d[key] = [
                try_number(item)
                if isinstance(item, str)
                else convert_strings_to_numbers(item)
                if isinstance(item, dict)
                else item
                for item in value
            ]
    return d
