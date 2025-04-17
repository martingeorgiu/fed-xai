import numpy as np
import xgboost as xgb
from rulecosi import Rule, RuleCOSIClassifier, RuleSet
from sklearn.base import check_array
from sklearn.metrics import classification_report

from fed_xai.data_loaders.loader import load_data
from fed_xai.federation.xgboost.const import class_names
from fed_xai.helpers.rulecosi_helpers import create_empty_rulecosi


def combining_rulecosi_explainer(clf: xgb.XGBClassifier) -> None:
    X_train_all, X_test_all, y_train_all, y_test_all = load_data(0, 1)

    X_train1, X_test1, y_train1, y_test1 = load_data(0, 2, withGlobal=False)
    # clf.save_model()
    rc1 = RuleCOSIClassifier(
        base_ensemble=clf,
        metric="f1",
        random_state=0,
        column_names=class_names,
    )
    rc1.fit(X_train1, y_train1)

    X_train2, X_test2, y_train2, y_test2 = load_data(1, 2, withGlobal=False)
    print(X_train1.columns)
    print(X_train1.head())
    print(X_train2.head())
    rc2 = RuleCOSIClassifier(
        base_ensemble=clf,
        metric="f1",
        random_state=0,
        column_names=class_names,
    )
    rc2.fit(X_train2, y_train2)
    # classes_ = unique_labels(y_train2)

    # Using combiner "trained on all data" does not change anything for the combining
    # rc_all = RuleCOSIClassifier(
    #     base_ensemble=clf,
    #     metric="f1",
    #     sort_by_class=None,
    #     random_state=0,
    #     column_names=X_train_all.columns,
    # )
    # rc_all.fit(X_train_all, y_train_all)

    rc_combiner = create_empty_rulecosi()

    rules1 = rc1.simplified_ruleset_
    rules2 = rc2.simplified_ruleset_
    print("== Simplified rules1 ==")
    rules1.print_rules()
    print("== Simplified rules2 ==")
    rules2.print_rules()
    # print("== Simplified rules using rc_all ==")
    # final_set1 = combine_rulesets(rc_all, rules1, rules2)
    # final_set1.print_rules()
    print("== Simplified rules using no data ==")
    final_set2 = combine_rulesets(rc_combiner, rules1, rules1)
    # pickle.dump(final_set2, open("final_set2.pkl", "wb"))
    final_set2.print_rules()

    # y_pred_rc_all = rc_all.predict(X_test_all)
    # print("====== rc_all Classification performance of XGBoost ======")
    # print(classification_report(y_test_all, y_pred_rc_all, digits=4))

    # X_test_all[0]
    X_test_all = check_array(X_test_all)

    # y_pred_rc_combiner = final_set2.predict()
    y_pred_rc_combiner = final_set2.predict(X_test_all)
    print("====== RC Combiner Classification performance of XGBoost ======")
    print(classification_report(y_test_all, y_pred_rc_combiner, digits=4))


def get_n_rules(rulesets: RuleSet | list | None) -> int:
    if rulesets is None:
        return 0
    n_rules = 0
    for ruleset in rulesets:
        for _ in ruleset:
            n_rules += 1
    return n_rules


def combine_rulesets(self: RuleCOSIClassifier, ruleset1, ruleset2) -> RuleSet:  # noqa: ANN001
    ruleset = _combine_rulesets(self, ruleset1, ruleset2)
    self._add_default_rule(ruleset)
    # self.simplified_ruleset_.prune_condition_map()
    self.simplified_ruleset_ = ruleset
    return ruleset


def _combine_rulesets(self: RuleCOSIClassifier, ruleset1, ruleset2):  # noqa: ANN001, ANN202
    """Combine all the rules belonging to ruleset1 and ruleset2 using
    the procedure described in the paper [ref]

    Main guiding procedure for combining rulesets for classification,
    make a combination of each of the class with itself and all the other
    classes

    :param ruleset1: First ruleset to be combined

    :param ruleset2: Second ruleset to be combiuned

    :return: ruleset containing the combination of ruleset1 and ruleset 2
    """
    combined_rules_set = set()
    for class_one in self.classes_:
        for class_two in self.classes_:
            s_ruleset1 = [rule1 for rule1 in ruleset1 if (rule1.y == [class_one])]
            s_ruleset2 = [rule2 for rule2 in ruleset2 if (rule2.y == [class_two])]
            # print("iteration")
            # print(s_ruleset1)
            # print(s_ruleset2)
            res = _combine_sliced_rulesets(self, s_ruleset1, s_ruleset2)
            # print("res")
            # print(res)
            combined_rules_set.update(res)

    # print("\n\ncombined_rules\n")
    # print(combined_rules)
    combined_rules = RuleSet(
        list(combined_rules_set), self._global_condition_map, classes=self.classes_
    )
    self._sort_ruleset(combined_rules)
    return combined_rules


def _combine_sliced_rulesets(self: RuleCOSIClassifier, s_ruleset1, s_ruleset2):  # noqa: ANN001, ANN202
    """Actual combination procedure between to class-sliced rulesets

    :param s_ruleset1: sliced ruleset 1 according to a class

    :param s_ruleset2: sliced ruleset according to a class

    :return: a set of rules containing the combined rules of s_ruleset1 and
     s_ruleset2
    """
    combined_rules = set()

    for r1 in s_ruleset1:
        for r2 in s_ruleset2:
            if len(r1.A) == 0 or len(r2.A) == 0:
                continue
            heuristics_dict = self._rule_heuristics.combine_heuristics(
                r1.heuristics_dict, r2.heuristics_dict
            )
            # print("Rule combining")
            # print(r1)
            # print(r2)
            # print(heuristics_dict)
            r1_AUr2_A = set(r1.A.union(r2.A))
            # print(r1_AUr2_A)
            if heuristics_dict["cov"] == 0:
                self._bad_combinations.add(frozenset(r1_AUr2_A))
                print("Ulalalal   1")
                continue

            if frozenset(r1_AUr2_A) in self._bad_combinations:
                print("Ulalalal   2")
                continue

            self.n_combinations_ += 1  # count the actual
            # number of combinations

            # create the new rule and compute class distribution
            # and predicted class
            weight = None

            if self._weights is None:
                ens_class_dist = np.mean([r1.ens_class_dist, r2.ens_class_dist], axis=0).reshape(
                    (len(self.classes_),)
                )
            else:
                ens_class_dist = np.average(
                    [r1.ens_class_dist, r2.ens_class_dist], axis=0, weights=[r1.weight, r2.weight]
                ).reshape((len(self.classes_),))
                weight = (r1.weight() + r2.weight) / 2
            logit_score = 0

            class_dist = ens_class_dist
            y_class_index = np.argmax(class_dist).item()
            y = np.array([self.classes_[y_class_index]])

            new_rule = Rule(
                frozenset(r1_AUr2_A),
                class_dist=class_dist,
                ens_class_dist=ens_class_dist,
                local_class_dist=ens_class_dist,
                # rule_class_dist,
                logit_score=logit_score,
                y=y,
                y_class_index=y_class_index,
                classes=self.classes_,
                weight=weight,
            )
            # print(f"new Rule: {new_rule}")

            if new_rule in self._good_combinations:
                # print("Ulalalal  - good")
                heuristics_dict = self._good_combinations[new_rule]
                new_rule.set_heuristics(heuristics_dict)
                combined_rules.add(new_rule)
            else:
                # print("Ulalalal  - not good")
                new_rule.set_heuristics(heuristics_dict)
                if new_rule.conf > self.conf_threshold and new_rule.cov > self.cov_threshold:
                    # print("Ulalalal  - not good 1")
                    combined_rules.add(new_rule)
                    self._good_combinations[new_rule] = heuristics_dict
                else:
                    # print("Ulalalal  - not good 2")
                    self._bad_combinations.add(frozenset(r1_AUr2_A))

    return combined_rules
