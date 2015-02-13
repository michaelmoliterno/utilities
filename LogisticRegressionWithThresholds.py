__author__ = 'michaelmoliterno'

from sklearn.linear_model import LogisticRegression


class LogisticRegressionWithThresholds(LogisticRegression):
    """
    This class is mostly identical to sklearn.linear_model.LogisticRegression (documented here):
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    This class adds an attribute threshold that allows for tuning of the threshold that predict
    uses to set y-predict to 1 or 0.  The default is 0 (same as LogisticRegression), but can be tuned
    to optimize accuracy, precision, recall, etc.

    """


    def __init__(self, threshold=0.5):
        super(LogisticRegressionWithThresholds, self).__init__()
        self.threshold = threshold

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        proba[proba >= self.threshold] = 1
        proba[proba < 1] = 0
        return proba.astype(int)