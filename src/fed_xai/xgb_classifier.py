import json
import operator

import numpy as np
from rulecosi import RuleSet
from rulecosi.rule_extraction import GBMClassifierRuleExtractor
from rulecosi.rule_heuristics import RuleHeuristics
def calc_rules(original_rulesets):
     processed_rulesets = np.copy.deepcopy(original_rulesets)

        # We create the heuristics object which will compute all the
        # heuristics related measures
        self._rule_heuristics = RuleHeuristics(X=self.X_, y=self.y_,
                                               classes_=self.classes_,
                                               condition_map=
                                               self._global_condition_map,
                                               cov_threshold=self.cov_threshold,
                                               conf_threshold=
                                               self.conf_threshold)
        if self.verbose > 0:
            print(f'Initializing sets and computing condition map...')
        self._initialize_sets()

        if str(
                self.base_ensemble_.__class__) == \
                "<class 'catboost.core.CatBoostClassifier'>":
            for ruleset in processed_rulesets:
                for rule in ruleset:
                    new_A = self._remove_opposite_conditions(set(rule.A),
                                                             rule.class_index)
                    rule.A = new_A

        for ruleset in processed_rulesets:
            self._rule_heuristics.compute_rule_heuristics(
                ruleset, recompute=True)
        self.simplified_ruleset_ = processed_rulesets[0]

        self._simplify_rulesets(
            self.simplified_ruleset_)
        y_pred = self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.compute_class_perf_fast(y_pred,
                                                         self.y_,
                                                         self.metric)
        self.simplified_ruleset_.rules.pop()

        self.n_combinations_ = 0

        self._early_stop_cnt = 0
        if self.early_stop > 0:
            early_stop = int(len(processed_rulesets) * self.early_stop)
        else:
            early_stop = len(processed_rulesets)

        if self.verbose > 0:
            print(f'Start combination process...')
            if self.verbose > 1:
                print(
                    f'Iteration {0}, Rule size: '
                    f'{len(self.simplified_ruleset_.rules)}, '
                    f'{self.metric}: '
                    f'{self.simplified_ruleset_.metric(self.metric)}')
        for i in range(1, len(processed_rulesets)):
            # combine the rules
            combined_rules = self._combine_rulesets(self.simplified_ruleset_,
                                                    processed_rulesets[i])

            if self.verbose > 1:
                print(f'Iteration{i}:')
                print(
                    f'\tCombined rules size: {len(combined_rules.rules)} rules')
            # prune inaccurate rules
            self._sequential_covering_pruning(combined_rules)
            if self.verbose > 1:
                print(
                    f'\tSequential covering pruned rules size: '
                    f'{len(combined_rules.rules)} rules')
            # simplify rules
            self._simplify_rulesets(combined_rules)
            if self.verbose > 1:
                print(
                    f'\tSimplified rules size: '
                    f'{len(combined_rules.rules)} rules')

            # skip if the combined rules are empty
            if len(combined_rules.rules) == 0:
                if self.verbose > 1:
                    print(f'\tCombined rules are empty, skipping iteration.')
                continue
            self.simplified_ruleset_, best_ruleset = self._evaluate_combinations(
                self.simplified_ruleset_, combined_rules)

            if self._early_stop_cnt >= early_stop:
                break
            if self.simplified_ruleset_.metric() == 1:
                break

        self.simplified_ruleset_.rules[:] = [rule for rule in
                                             self.simplified_ruleset_.rules
                                             if rule.cov > 0]
        if self.verbose > 0:
            print(f'Finish combination process, adding default rule...')

        _ = self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.prune_condition_map()
        end_time = time.time()
        self.combination_time_ = end_time - start_time
        if self.verbose > 0:
            print(
                f'R size: {len(self.simplified_ruleset_.rules)}, {self.metric}:'
                f' {self.simplified_ruleset_.metric(self.metric)}')
class XGBClassifierExtractorForDebug(GBMClassifierRuleExtractor):
    """Rule extraction for a Gradient Boosting Tree ensemble classifier.
    This class accept only XGB implementation

    Parameters
    ----------
    base_ensemble: BaseEnsemble object, default = None
        A BaseEnsemble estimator object. The supported types are:
            - :class:`xgboost.XGBClassifier`

    column_names: array of string, default=None Array of strings with the
    name of the columns in the data. This is useful for displaying the name
    of the features in the generated rules.

    classes: ndarray, shape (n_classes,)
        The classes seen when fitting the ensemble.

    X: array-like, shape (n_samples, n_features)
        The training input samples.

    """

    def extract_rules(self):
        """Main method for extracting the rules of tree ensembles

        :return: an array of :class:`rulecosi.rules.RuleSet'
        """
        rulesets = []
        global_condition_map = dict()
        booster = self._ensemble.get_booster()
        booster.feature_names = self._column_names.to_list()
        xgb_tree_dicts = booster.get_dump(dump_format="json")
        n_nodes = (
            booster.trees_to_dataframe()[["Tree", "Node"]]
            .groupby("Tree")
            .count()
            .to_numpy()
        )
        if len(self.classes_) > 2:
            # t_idx = 0
            for tid in range(self._ensemble.n_estimators):
                current_tree_rules = []
                current_tree_condition_map = dict()
                for cid, _ in enumerate(self.classes_):
                    original_ruleset = self.get_base_ruleset(
                        self.get_tree_dict(xgb_tree_dicts[tid], n_nodes[tid]),
                        class_index=cid,
                        tree_index=tid,
                    )
                    current_tree_rules.extend(original_ruleset.rules)
                    current_tree_condition_map.update(original_ruleset.condition_map)
                    # t_idx += 1
                current_tree_ruleset = RuleSet(
                    current_tree_rules,
                    current_tree_condition_map,
                    classes=self.classes_,
                )
                global_condition_map.update(current_tree_ruleset.condition_map)
                rulesets.append(current_tree_ruleset)
            return rulesets, global_condition_map

        else:  # binary classification

            for t_idx, xgb_t_dict in enumerate(xgb_tree_dicts):
                original_ruleset = self.get_base_ruleset(
                    self.get_tree_dict(xgb_t_dict, n_nodes[t_idx]),
                    class_index=0,
                    tree_index=t_idx,
                )
                rulesets.append(original_ruleset)
                global_condition_map.update(original_ruleset.condition_map)
            return rulesets, global_condition_map

    # def _get_class_dist(self, raw_to_proba):
    #     return np.array([raw_to_proba.item(), 1 - raw_to_proba.item()])

    def get_tree_dict(self, base_tree, n_nodes=0):
        """Create a dictionary with the information inside the base_tree

        :param base_tree: :class: `sklearn.tree.Tree` object wich is an array
        representation of a tree

        :param n_nodes: number of nodes in the tree

        :return: a dictionary conatining the information of the base_tree
        """
        tree_dict = {
            "children_left": np.full(n_nodes, fill_value=-1),
            "children_right": np.full(n_nodes, fill_value=-1),
            "feature": np.full(n_nodes, fill_value=0),
            "threshold": np.full(n_nodes, fill_value=0.0),
            "value": np.full(n_nodes, fill_value=0.0),
            "n_samples": np.full(n_nodes, fill_value=-1),
            "n_nodes": n_nodes,
        }

        tree = json.loads(base_tree)
        self._populate_tree_dict(tree, tree_dict)
        return tree_dict

    def _populate_tree_dict(self, tree, tree_dict):
        """Populate the tree dictionary specifically for this type of GBM
        implementation. This is needed because each GBM implementation output
        the trees in different formats

        :param tree: the current tree to be used as a source

        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        """
        node_id = tree["nodeid"]
        if "leaf" in tree:
            tree_dict["value"][node_id] = tree["leaf"]
            return
        if "children" in tree:
            tree_dict["children_left"][node_id] = tree["children"][0]["nodeid"]
            tree_dict["children_right"][node_id] = tree["children"][1]["nodeid"]
            # tree_dict['feature'][node_id] = int(tree['split'][1:])
            # if not str.isdigit(tree['split']):
            # new change 2023/03/14 . setting feature names before
            # dumping allows to ensure feature names are preserved from
            # self.column_names
            tree_dict["feature"][node_id] = np.where(
                self._column_names == tree["split"]
            )[
                0
            ].item()  # 2021/23/06 change, the split directly

            # tree_dict['feature'][node_id] = \
            #     int(tree['split'].replace('f', ''))
            # 2022/04/16 change obtain directly the feature index
            # np.where(self._column_names == tree['split'])[
            #     0].item()  # 2021/23/06 change, the split directly

            # print('feature: ', tree['split'])
            # print('node_id: ', tree_dict['feature'][node_id])
            # else:
            #     tree_dict['feature'][node_id] = int(
            #         tree['split'])  # 2021/23/06 change, the split directly
            tree_dict["threshold"][node_id] = tree["split_condition"]
            self._populate_tree_dict(tree["children"][0], tree_dict)
            self._populate_tree_dict(tree["children"][1], tree_dict)

    def get_split_operators(self):
        """Return the operator applied for the left and right branches of
        the tree. This function is needed because different implementations
        of trees use different operators for the children nodes.

        :return: a tuple containing the left and right operator used for
        creating conditions
        """
        op_left = operator.lt  # Operator.LESS_THAN
        op_right = operator.ge  # Operator.GREATER_OR_EQUAL_THAN
        return op_left, op_right

    def _get_gbm_init(self):
        """get the initial estimate of a GBM ensemble

        :return: a double value of the initial estimate of the GBM ensemble
        """
        if self._ensemble.base_score is None:
            return self.class_ratio
        else:
            return self._ensemble.base_score


class XGBClassifierExtractorForBooster(GBMClassifierRuleExtractor):
    """Rule extraction for a Gradient Boosting Tree ensemble classifier.
    This class accept only XGB implementation

    Parameters
    ----------
    base_ensemble: BaseEnsemble object, default = None
        A BaseEnsemble estimator object. The supported types are:
            - :class:`xgboost.XGBClassifier`

    column_names: array of string, default=None Array of strings with the
    name of the columns in the data. This is useful for displaying the name
    of the features in the generated rules.

    classes: ndarray, shape (n_classes,)
        The classes seen when fitting the ensemble.

    X: array-like, shape (n_samples, n_features)
        The training input samples.

    """

    def extract_rules(self):
        """Main method for extracting the rules of tree ensembles

        :return: an array of :class:`rulecosi.rules.RuleSet'
        """
        rulesets = []
        global_condition_map = dict()
        booster = self._ensemble
        # TODO: revert
        # booster = self._ensemble.get_booster()
        booster.feature_names = self._column_names
        # booster.feature_names = self._column_names.to_list()

        dump_list = booster.get_dump(dump_format="json")
        num_trees = len(dump_list)

        xgb_tree_dicts = booster.get_dump(dump_format="json")
        n_nodes = (
            booster.trees_to_dataframe()[["Tree", "Node"]]
            .groupby("Tree")
            .count()
            .to_numpy()
        )
        if len(self.classes_) > 2:
            # ct_list = [(cid, tid)  # class tree pair list
            #            for tid in [0, 1, 2]
            #            for cid in self.classes_]
            # for (t_idx, (xgb_t_dict, (cid, tid))) in enumerate(
            #         zip(xgb_tree_dicts, ct_list)):
            t_idx = 0
            # for tid in range(self._ensemble.n_estimators):
            for tid in range(num_trees):
                current_tree_rules = []
                current_tree_condition_map = dict()
                for cid, _ in enumerate(self.classes_):
                    # booster_df = booster.trees_to_dataframe()
                    # n_nodes = booster_df[booster_df['Tree'] == t_idx][
                    #     ['Tree', 'Node']].groupby(
                    #     'Tree').count().to_numpy().item()
                    original_ruleset = self.get_base_ruleset(
                        self.get_tree_dict(xgb_tree_dicts[t_idx], n_nodes[t_idx]),
                        class_index=cid,
                        tree_index=tid,
                    )
                    current_tree_rules.extend(original_ruleset.rules)
                    current_tree_condition_map.update(original_ruleset.condition_map)
                    t_idx += 1
                current_tree_ruleset = RuleSet(
                    current_tree_rules,
                    current_tree_condition_map,
                    classes=self.classes_,
                )
                global_condition_map.update(current_tree_ruleset.condition_map)
                rulesets.append(current_tree_ruleset)
            return rulesets, global_condition_map

        else:  # binary classification

            for t_idx, xgb_t_dict in enumerate(xgb_tree_dicts):
                original_ruleset = self.get_base_ruleset(
                    self.get_tree_dict(xgb_t_dict, n_nodes[t_idx]),
                    class_index=0,
                    tree_index=t_idx,
                )
                rulesets.append(original_ruleset)
                global_condition_map.update(original_ruleset.condition_map)
            return rulesets, global_condition_map

    # def _get_class_dist(self, raw_to_proba):
    #     return np.array([raw_to_proba.item(), 1 - raw_to_proba.item()])

    def get_tree_dict(self, base_tree, n_nodes=0):
        """Create a dictionary with the information inside the base_tree

        :param base_tree: :class: `sklearn.tree.Tree` object wich is an array
        representation of a tree

        :param n_nodes: number of nodes in the tree

        :return: a dictionary conatining the information of the base_tree
        """
        tree_dict = {
            "children_left": np.full(n_nodes, fill_value=-1),
            "children_right": np.full(n_nodes, fill_value=-1),
            "feature": np.full(n_nodes, fill_value=0),
            "threshold": np.full(n_nodes, fill_value=0.0),
            "value": np.full(n_nodes, fill_value=0.0),
            "n_samples": np.full(n_nodes, fill_value=-1),
            "n_nodes": n_nodes,
        }

        tree = json.loads(base_tree)
        self._populate_tree_dict(tree, tree_dict)
        return tree_dict

    def _populate_tree_dict(self, tree, tree_dict):
        """Populate the tree dictionary specifically for this type of GBM
        implementation. This is needed because each GBM implementation output
        the trees in different formats

        :param tree: the current tree to be used as a source

        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        """
        node_id = tree["nodeid"]
        if "leaf" in tree:
            tree_dict["value"][node_id] = tree["leaf"]
            return
        if "children" in tree:
            tree_dict["children_left"][node_id] = tree["children"][0]["nodeid"]
            tree_dict["children_right"][node_id] = tree["children"][1]["nodeid"]
            # tree_dict['feature'][node_id] = int(tree['split'][1:])
            # if not str.isdigit(tree['split']):
            # new change 2023/03/14 . setting feature names before
            # dumping allows to ensure feature names are preserved from
            # self.column_names
            print(tree)
            print(tree_dict)
            # indices = np.nonzero(np.atleast_1d(self._column_names == tree["split"]))

            # Check that there is exactly one match.
            # if indices.size != 1:
            # if True:
            #     tt = tree["split"] in self._column_names
            #     raise ValueError(
            #         f"Expected exactly one match for feature name '{tree['split']}', found {len(indices[0])}, {self._column_names}, {tt}"
            #     )

            # Assign the single matching index.
            tree_dict["feature"][node_id] = self._column_names.index(tree["split"])
            # tree_dict["feature"][node_id] = np.where(
            #     self._column_names == tree["split"]
            # )[
            #     0
            # ].item()  # 2021/23/06 change, the split directly

            # tree_dict['feature'][node_id] = \
            #     int(tree['split'].replace('f', ''))
            # 2022/04/16 change obtain directly the feature index
            # np.where(self._column_names == tree['split'])[
            #     0].item()  # 2021/23/06 change, the split directly

            # print('feature: ', tree['split'])
            # print('node_id: ', tree_dict['feature'][node_id])
            # else:
            #     tree_dict['feature'][node_id] = int(
            #         tree['split'])  # 2021/23/06 change, the split directly
            tree_dict["threshold"][node_id] = tree["split_condition"]
            self._populate_tree_dict(tree["children"][0], tree_dict)
            self._populate_tree_dict(tree["children"][1], tree_dict)

    def get_split_operators(self):
        """Return the operator applied for the left and right branches of
        the tree. This function is needed because different implementations
        of trees use different operators for the children nodes.

        :return: a tuple containing the left and right operator used for
        creating conditions
        """
        op_left = operator.lt  # Operator.LESS_THAN
        op_right = operator.ge  # Operator.GREATER_OR_EQUAL_THAN
        return op_left, op_right

    def _get_gbm_init(self):
        """get the initial estimate of a GBM ensemble

        :return: a double value of the initial estimate of the GBM ensemble
        """
        return 8.547718e-1
        # if self._ensemble.base_score is None:
        #     return self.class_ratio
        # else:
        #     return self._ensemble.base_score
