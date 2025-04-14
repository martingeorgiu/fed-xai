import xgboost as xgb
from rulecosi import RuleCOSIClassifier, RuleSet
from sklearn.metrics import classification_report, roc_auc_score

from fed_xai.data_loaders.loader import load_data_with_smote


def rulecosi_explainer(clf: xgb.XGBClassifier) -> None:
    X_train, X_test, y_train, y_test = load_data_with_smote(0, 1)

    rc = RuleCOSIClassifier(
        base_ensemble=clf,
        metric="f1",
        # n_estimators=100,
        # tree_max_depth=3,
        # conf_threshold=0.9,
        # cov_threshold=0.0,
        random_state=0,
        column_names=X_train.columns,
    )
    rc.fit(X_train, y_train)

    print("== Original XGBoost ensemble ==")
    print(f"Number of trees: {rc.base_ensemble_.n_estimators} trees")
    print(f"Number of rules: {get_n_rules(rc.original_rulesets_)} rules\n")

    print("== Simplified rules ==")
    rc.simplified_ruleset_.print_rules()

    y_pred = rc.predict(X_test)
    y_pred_ens = rc.base_ensemble_.predict(X_test, validate_features=False)

    print(f"Combinations: {rc.n_combinations_}")
    print(f"Time: {rc.combination_time_}\n")
    print("====== Classification performance of XGBoost ======")
    print(classification_report(y_test, y_pred_ens, digits=4))
    print(roc_auc_score(y_test, y_pred_ens))
    print("\n====== Classification performance of simplified rules ======")
    print(classification_report(y_test, y_pred, digits=4))
    print(roc_auc_score(y_test, y_pred))
    print("\n")


def get_n_rules(rulesets: RuleSet | list | None) -> int:
    if rulesets is None:
        return 0
    n_rules = 0
    for ruleset in rulesets:
        for _ in ruleset:
            n_rules += 1
    return n_rules


# class XGBClassifierExtractorForDebug(GBMClassifierRuleExtractor):
#     """Rule extraction for a Gradient Boosting Tree ensemble classifier.
#     This class accept only XGB implementation

#     Parameters
#     ----------
#     base_ensemble: BaseEnsemble object, default = None
#         A BaseEnsemble estimator object. The supported types are:
#             - :class:`xgboost.XGBClassifier`

#     column_names: array of string, default=None Array of strings with the
#     name of the columns in the data. This is useful for displaying the name
#     of the features in the generated rules.

#     classes: ndarray, shape (n_classes,)
#         The classes seen when fitting the ensemble.

#     X: array-like, shape (n_samples, n_features)
#         The training input samples.

#     """

#     def extract_rules(self):
#         """Main method for extracting the rules of tree ensembles

#         :return: an array of :class:`rulecosi.rules.RuleSet'
#         """
#         rulesets = []
#         global_condition_map = dict()
#         booster = self._ensemble.get_booster()
#         booster.feature_names = self._column_names.to_list()
#         xgb_tree_dicts = booster.get_dump(dump_format="json")
#         n_nodes = booster.trees_to_dataframe()[["Tree", "Node"]].groupby("Tree").count()
#               .to_numpy()
#         if len(self.classes_) > 2:
#             # t_idx = 0
#             for tid in range(self._ensemble.n_estimators):
#                 current_tree_rules = []
#                 current_tree_condition_map = dict()
#                 for cid, _ in enumerate(self.classes_):
#                     original_ruleset = self.get_base_ruleset(
#                         self.get_tree_dict(xgb_tree_dicts[tid], n_nodes[tid]),
#                         class_index=cid,
#                         tree_index=tid,
#                     )
#                     current_tree_rules.extend(original_ruleset.rules)
#                     current_tree_condition_map.update(original_ruleset.condition_map)
#                     # t_idx += 1
#                 current_tree_ruleset = RuleSet(
#                     current_tree_rules,
#                     current_tree_condition_map,
#                     classes=self.classes_,
#                 )
#                 global_condition_map.update(current_tree_ruleset.condition_map)
#                 rulesets.append(current_tree_ruleset)
#             return rulesets, global_condition_map

#         else:  # binary classification
#             for t_idx, xgb_t_dict in enumerate(xgb_tree_dicts):
#                 original_ruleset = self.get_base_ruleset(
#                     self.get_tree_dict(xgb_t_dict, n_nodes[t_idx]),
#                     class_index=0,
#                     tree_index=t_idx,
#                 )
#                 rulesets.append(original_ruleset)
#                 global_condition_map.update(original_ruleset.condition_map)
#             return rulesets, global_condition_map

#     # def _get_class_dist(self, raw_to_proba):
#     #     return np.array([raw_to_proba.item(), 1 - raw_to_proba.item()])

#     def get_tree_dict(self, base_tree, n_nodes=0):
#         """Create a dictionary with the information inside the base_tree

#         :param base_tree: :class: `sklearn.tree.Tree` object wich is an array
#         representation of a tree

#         :param n_nodes: number of nodes in the tree

#         :return: a dictionary conatining the information of the base_tree
#         """
#         tree_dict = {
#             "children_left": np.full(n_nodes, fill_value=-1),
#             "children_right": np.full(n_nodes, fill_value=-1),
#             "feature": np.full(n_nodes, fill_value=0),
#             "threshold": np.full(n_nodes, fill_value=0.0),
#             "value": np.full(n_nodes, fill_value=0.0),
#             "n_samples": np.full(n_nodes, fill_value=-1),
#             "n_nodes": n_nodes,
#         }

#         tree = json.loads(base_tree)
#         self._populate_tree_dict(tree, tree_dict)
#         return tree_dict

#     def _populate_tree_dict(self, tree, tree_dict):
#         """Populate the tree dictionary specifically for this type of GBM
#         implementation. This is needed because each GBM implementation output
#         the trees in different formats

#         :param tree: the current tree to be used as a source

#         :param tree_dict: a dictionary containing  the information of the
#         base_tree (arrays on :class: `sklearn.tree.Tree` class

#         """
#         node_id = tree["nodeid"]
#         if "leaf" in tree:
#             tree_dict["value"][node_id] = tree["leaf"]
#             return
#         if "children" in tree:
#             tree_dict["children_left"][node_id] = tree["children"][0]["nodeid"]
#             tree_dict["children_right"][node_id] = tree["children"][1]["nodeid"]
#             # tree_dict['feature'][node_id] = int(tree['split'][1:])
#             # if not str.isdigit(tree['split']):
#             # new change 2023/03/14 . setting feature names before
#             # dumping allows to ensure feature names are preserved from
#             # self.column_names
#             tree_dict["feature"][node_id] = np.where(self._column_names == tree["split"])[
#                 0
#             ].item()  # 2021/23/06 change, the split directly

#             # tree_dict['feature'][node_id] = \
#             #     int(tree['split'].replace('f', ''))
#             # 2022/04/16 change obtain directly the feature index
#             # np.where(self._column_names == tree['split'])[
#             #     0].item()  # 2021/23/06 change, the split directly

#             # print('feature: ', tree['split'])
#             # print('node_id: ', tree_dict['feature'][node_id])
#             # else:
#             #     tree_dict['feature'][node_id] = int(
#             #         tree['split'])  # 2021/23/06 change, the split directly
#             tree_dict["threshold"][node_id] = tree["split_condition"]
#             self._populate_tree_dict(tree["children"][0], tree_dict)
#             self._populate_tree_dict(tree["children"][1], tree_dict)

#     def get_split_operators(self):
#         """Return the operator applied for the left and right branches of
#         the tree. This function is needed because different implementations
#         of trees use different operators for the children nodes.

#         :return: a tuple containing the left and right operator used for
#         creating conditions
#         """
#         op_left = operator.lt  # Operator.LESS_THAN
#         op_right = operator.ge  # Operator.GREATER_OR_EQUAL_THAN
#         return op_left, op_right

#     def _get_gbm_init(self):
#         """get the initial estimate of a GBM ensemble

#         :return: a double value of the initial estimate of the GBM ensemble
#         """
#         if self._ensemble.base_score is None:
#             return self.class_ratio
#         else:
#             return self._ensemble.base_score


# class XGBClassifierExtractorForBooster(GBMClassifierRuleExtractor):
#     """Rule extraction for a Gradient Boosting Tree ensemble classifier.
#     This class accept only XGB implementation

#     Parameters
#     ----------
#     base_ensemble: BaseEnsemble object, default = None
#         A BaseEnsemble estimator object. The supported types are:
#             - :class:`xgboost.XGBClassifier`

#     column_names: array of string, default=None Array of strings with the
#     name of the columns in the data. This is useful for displaying the name
#     of the features in the generated rules.

#     classes: ndarray, shape (n_classes,)
#         The classes seen when fitting the ensemble.

#     X: array-like, shape (n_samples, n_features)
#         The training input samples.

#     """

#     def extract_rules(self):
#         """Main method for extracting the rules of tree ensembles

#         :return: an array of :class:`rulecosi.rules.RuleSet'
#         """
#         rulesets = []
#         global_condition_map = dict()
#         booster = self._ensemble
#         # TODO: revert
#         # booster = self._ensemble.get_booster()
#         booster.feature_names = self._column_names
#         # booster.feature_names = self._column_names.to_list()

#         dump_list = booster.get_dump(dump_format="json")
#         num_trees = len(dump_list)

#         xgb_tree_dicts = booster.get_dump(dump_format="json")
#         n_nodes = booster.trees_to_dataframe()[["Tree", "Node"]].groupby("Tree").count()
# .to_numpy()
#         if len(self.classes_) > 2:
#             # ct_list = [(cid, tid)  # class tree pair list
#             #            for tid in [0, 1, 2]
#             #            for cid in self.classes_]
#             # for (t_idx, (xgb_t_dict, (cid, tid))) in enumerate(
#             #         zip(xgb_tree_dicts, ct_list)):
#             t_idx = 0
#             # for tid in range(self._ensemble.n_estimators):
#             for tid in range(num_trees):
#                 current_tree_rules = []
#                 current_tree_condition_map = dict()
#                 for cid, _ in enumerate(self.classes_):
#                     # booster_df = booster.trees_to_dataframe()
#                     # n_nodes = booster_df[booster_df['Tree'] == t_idx][
#                     #     ['Tree', 'Node']].groupby(
#                     #     'Tree').count().to_numpy().item()
#                     original_ruleset = self.get_base_ruleset(
#                         self.get_tree_dict(xgb_tree_dicts[t_idx], n_nodes[t_idx]),
#                         class_index=cid,
#                         tree_index=tid,
#                     )
#                     current_tree_rules.extend(original_ruleset.rules)
#                     current_tree_condition_map.update(original_ruleset.condition_map)
#                     t_idx += 1
#                 current_tree_ruleset = RuleSet(
#                     current_tree_rules,
#                     current_tree_condition_map,
#                     classes=self.classes_,
#                 )
#                 global_condition_map.update(current_tree_ruleset.condition_map)
#                 rulesets.append(current_tree_ruleset)
#             return rulesets, global_condition_map

#         else:  # binary classification
#             for t_idx, xgb_t_dict in enumerate(xgb_tree_dicts):
#                 original_ruleset = self.get_base_ruleset(
#                     self.get_tree_dict(xgb_t_dict, n_nodes[t_idx]),
#                     class_index=0,
#                     tree_index=t_idx,
#                 )
#                 rulesets.append(original_ruleset)
#                 global_condition_map.update(original_ruleset.condition_map)
#             return rulesets, global_condition_map

#     # def _get_class_dist(self, raw_to_proba):
#     #     return np.array([raw_to_proba.item(), 1 - raw_to_proba.item()])

#     def get_tree_dict(self, base_tree, n_nodes=0):
#         """Create a dictionary with the information inside the base_tree

#         :param base_tree: :class: `sklearn.tree.Tree` object wich is an array
#         representation of a tree

#         :param n_nodes: number of nodes in the tree

#         :return: a dictionary conatining the information of the base_tree
#         """
#         tree_dict = {
#             "children_left": np.full(n_nodes, fill_value=-1),
#             "children_right": np.full(n_nodes, fill_value=-1),
#             "feature": np.full(n_nodes, fill_value=0),
#             "threshold": np.full(n_nodes, fill_value=0.0),
#             "value": np.full(n_nodes, fill_value=0.0),
#             "n_samples": np.full(n_nodes, fill_value=-1),
#             "n_nodes": n_nodes,
#         }

#         tree = json.loads(base_tree)
#         self._populate_tree_dict(tree, tree_dict)
#         return tree_dict

#     def _populate_tree_dict(self, tree, tree_dict):
#         """Populate the tree dictionary specifically for this type of GBM
#         implementation. This is needed because each GBM implementation output
#         the trees in different formats

#         :param tree: the current tree to be used as a source

#         :param tree_dict: a dictionary containing  the information of the
#         base_tree (arrays on :class: `sklearn.tree.Tree` class

#         """
#         node_id = tree["nodeid"]
#         if "leaf" in tree:
#             tree_dict["value"][node_id] = tree["leaf"]
#             return
#         if "children" in tree:
#             tree_dict["children_left"][node_id] = tree["children"][0]["nodeid"]
#             tree_dict["children_right"][node_id] = tree["children"][1]["nodeid"]
#             # tree_dict['feature'][node_id] = int(tree['split'][1:])
#             # if not str.isdigit(tree['split']):
#             # new change 2023/03/14 . setting feature names before
#             # dumping allows to ensure feature names are preserved from
#             # self.column_names
#             print(tree)
#             print(tree_dict)
#             # indices = np.nonzero(np.atleast_1d(self._column_names == tree["split"]))

#             # Check that there is exactly one match.
#             # if indices.size != 1:
#             # if True:
#             #     tt = tree["split"] in self._column_names
#             #     raise ValueError(
#             #         f"Expected exactly one match for feature name '{tree['split']}',
#  found {len(indices[0])}, {self._column_names}, {tt}"
#             #     )

#             # Assign the single matching index.
#             tree_dict["feature"][node_id] = self._column_names.index(tree["split"])
#             # tree_dict["feature"][node_id] = np.where(
#             #     self._column_names == tree["split"]
#             # )[
#             #     0
#             # ].item()  # 2021/23/06 change, the split directly

#             # tree_dict['feature'][node_id] = \
#             #     int(tree['split'].replace('f', ''))
#             # 2022/04/16 change obtain directly the feature index
#             # np.where(self._column_names == tree['split'])[
#             #     0].item()  # 2021/23/06 change, the split directly

#             # print('feature: ', tree['split'])
#             # print('node_id: ', tree_dict['feature'][node_id])
#             # else:
#             #     tree_dict['feature'][node_id] = int(
#             #         tree['split'])  # 2021/23/06 change, the split directly
#             tree_dict["threshold"][node_id] = tree["split_condition"]
#             self._populate_tree_dict(tree["children"][0], tree_dict)
#             self._populate_tree_dict(tree["children"][1], tree_dict)

#     def get_split_operators(self):
#         """Return the operator applied for the left and right branches of
#         the tree. This function is needed because different implementations
#         of trees use different operators for the children nodes.

#         :return: a tuple containing the left and right operator used for
#         creating conditions
#         """
#         op_left = operator.lt  # Operator.LESS_THAN
#         op_right = operator.ge  # Operator.GREATER_OR_EQUAL_THAN
#         return op_left, op_right

#     def _get_gbm_init(self):
#         """get the initial estimate of a GBM ensemble

#         :return: a double value of the initial estimate of the GBM ensemble
#         """
#         return 8.547718e-1
#         # if self._ensemble.base_score is None:
#         #     return self.class_ratio
#         # else:
#         #     return self._ensemble.base_score
