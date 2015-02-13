__author__ = 'michaelmoliterno'


class LogisticRegressionWithThresholds(LogisticRegression):
    def __init__(self, threshold=0.5):
        super(LogisticRegressionWithThresholds, self).__init__()
        self.threshold = threshold

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        proba[proba >= self.threshold] = 1
        proba[proba < 1] = 0
        return proba.astype(int)