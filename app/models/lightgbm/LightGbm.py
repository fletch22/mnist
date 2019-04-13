import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin


class LightGbm(BaseEstimator, ClassifierMixin):

    # def __init__(self):
    #     self.lgb =

    def fit(self, X, y):
        params = {}
        params['max_bin'] = 10
        params['learning_rate'] = 0.0021  # shrinkage_rate
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'regression'
        params['metric'] = 'l1'  # or 'mae'
        params['sub_feature'] = 0.345  # feature_fraction (small values => use very different submodels)
        params['bagging_fraction'] = 0.85  # sub_row
        params['bagging_freq'] = 40
        params['num_leaves'] = 512  # num_leaf
        params['min_data'] = 500  # min_data_in_leaf
        params['min_hessian'] = 0.05  # min_sum_hessian_in_leaf
        params['verbose'] = 0
        params['feature_fraction_seed'] = 2
        params['bagging_seed'] = 3

        d_train = lgb.Dataset(X, label=y)
        self.clf = lgb.train(params, d_train, 430)

    def predict(self, x_test):
        return self.clf.predict(x_test)

    def predict_proba(self, X, **kwargs):
        return self.clf.predict_proba(X, **kwargs)

