from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from app.keras import logistic_regression


class TestLogisticRegression(TestCase):

    def test_log_reg(self):
        # Arrange
        # Act
        logistic_regression.main()

from deslib.des.knora_e import KNORAE

# Train a pool of 10 classifiers
pool_classifiers = RandomForestClassifier(n_estimators=10)
pool_classifiers.fit(X_train, y_train)

# Initialize the DES model
knorae = KNORAE(pool_classifiers)

# Preprocess the Dynamic Selection dataset (DSEL)
knorae.fit(X_dsel, y_dsel)

# Predict new examples:
knorae.predict(X_test)