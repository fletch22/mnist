import os
from six.moves import urllib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from app.config import config

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

def log_reg():
    X = mnist["data"]

    # file_path = os.path.join(config.DATA_FOLDER_PATH, 'large_train.csv')

    y = mnist["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    result = log_reg.score(X_test, y_test)

    print(str(result))
    print(type(result))
