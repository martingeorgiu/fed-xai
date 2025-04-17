import pickle
from typing import cast

import pandas as pd
from rulecosi import RuleCOSIClassifier, RuleSet

from fed_xai.federation.xgboost.const import class_names


def create_empty_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Create an empty DataFrame and Series
    """
    number_of_rows = 1000
    X = pd.DataFrame(0, index=range(number_of_rows), columns=class_names)
    # The y data have to have same number of 0 and 1 and then we are golden!!!!!
    y = pd.Series([1 if i % 2 == 0 else 0 for i in range(number_of_rows)])

    return X, y


def create_empty_rulecosi() -> RuleCOSIClassifier:
    X, y = create_empty_data()

    rc_combiner = RuleCOSIClassifier(
        base_ensemble=None,
        metric="f1",
        sort_by_class=None,
        random_state=0,
        column_names=class_names,
    )
    rc_combiner.fit(X, y)

    return rc_combiner


def bytes_to_ruleset(bytes: bytes) -> RuleSet:
    """
    Convert bytes to RuleSet
    """
    rules = cast(RuleSet, pickle.loads(bytes))
    return rules


def ruleset_to_bytes(rules: RuleSet) -> bytes:
    """
    Convert RuleSet to bytes
    """
    rules_bytes = pickle.dumps(rules)
    return rules_bytes
