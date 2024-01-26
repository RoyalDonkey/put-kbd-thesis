"""
Wrapper class for the plot/create_results.py script, because xgboost does not
natively expose the fit() and predict() methods like sklearn.
"""
import numpy as np
import xgboost_classifier as xgbc
import xgboost as xgb
from typing import Self


class XGBoostResultsWrapper:
    def __init__(self):
        self.key2num = None
        self.num2key = None
        self.bst = None
        self.dtrain = None
        self.evals_result = None
        self.params = xgbc.configure() | {
            # Best params for default dataset, MFCC & fft
            # We assume the same hyperparameters for all other variants as well, because
            # it's computationally infeasible to optimize separately for every set of
            # train and tests datasets.
            "min_split_loss"    : 0.2,
            "min_child_weight"  : 0.2,
        }

    def fit(self, x: np.array, y: np.array) -> Self:
        """Train the model on examples x and labels y."""
        present_keys = list(set(y))
        present_keys.sort(key=ord)
        self.key2num = dict(zip(present_keys, range(len(present_keys))))
        self.num2key = dict(zip(range(len(present_keys)), present_keys))

        self.dtrain = xgb.DMatrix(x, label=np.array([self.key2num[v] for v in y]))
        self.bst = xgbc.train_model(self.dtrain, None, None, self.params)
        return self

    def predict(self, x: np.array) -> np.array:
        """Get predictions from the model."""
        assert self.num2key is not None, "use fit() first!"
        assert self.bst is not None, "use fit() first!"
        y = self.bst.predict(xgb.DMatrix(x))
        return np.array([self.num2key[v] for v in y])
