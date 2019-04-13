import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from torch import nn
import torch.nn.functional as F
import logging

from skorch import NeuralNetClassifier


class F22SkorchCnn(nn.Module):
    def __init__(self, num_units=128, num_features=None, nonlin=F.relu):
        super(F22SkorchCnn, self).__init__()

        self.dense0 = nn.Linear(num_features, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X


def try_it():
    X, y = make_classification(1000, 784, n_informative=10, random_state=0)

    logging.info('X: {}'.format(X.shape))

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    net = NeuralNetClassifier(
        F22SkorchCnn,
        module__num_features=X.shape[1],
        max_epochs=10,
        lr=0.1,
    )

    # kwargs = {'module__num_units': 777}
    # net.set_params(**kwargs)

    net.fit(X, y)
    y_proba = net.predict_proba(X)
    logging.info('y_proba: {}'.format(y_proba))

    scores = cross_val_score(net, X, y, cv=3, scoring='accuracy')
    logging.info('Score: {}'.format(scores))
    logging.info("Accuracy: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), 'F22SkorchCnn'))
