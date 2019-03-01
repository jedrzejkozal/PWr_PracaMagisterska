import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator

from Models.SimpleReNet.SimpleReNet import *


class ReNetWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self):
        w_p, h_p = 2, 2
        reNet_hidden_size = 1
        fully_conn_hidden_size = 1
        dropout = 0.2
        num_classes = 2

        self.model = SimpleReNet([[w_p, h_p]],
                reNet_hidden_size, fully_conn_hidden_size, num_classes)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['categorical_accuracy'])


    def fit(self, X, y=None):
        print("\n\nfit X arg: ", X)
        print("\n\nfit X.shape arg: ", X.shape)
        self.model.fit(X, y, epochs=1, shuffle=False)

        return self


    def predict(self, x):
        return self.model.predict(x)


    def score(self, X, y=None):
        return(sum(self.predict(X)))



@pytest.fixture
def sut():
    return ReNetWrapper()


@pytest.mark.skip(reason="test just to check what check_estimator does")
def test_sklearn_estimator_check(sut):
    check_estimator(sut)
    assert True


from sklearn.dummy import DummyClassifier

@pytest.mark.skip(reason="getting exception: Estimator doesn't check for NaN and inf in fit.")
def test_sklearn_dummy_classifier_estimator_check():
    sut = DummyClassifier()
    check_estimator(sut)
    assert True
