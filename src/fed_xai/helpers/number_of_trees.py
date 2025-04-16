import xgboost as xgb


def get_number_of_trees(bst: xgb.Booster) -> int:
    dump_list = bst.get_dump()
    num_trees = len(dump_list)
    return num_trees
