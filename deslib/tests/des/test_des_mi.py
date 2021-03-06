from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.linear_model import Perceptron
from sklearn.utils.estimator_checks import check_estimator

from deslib.des.des_mi import DESMI


def test_check_estimator():
    check_estimator(DESMI())


# TODO: create test routine for the estimate_competence method


@pytest.mark.parametrize('alpha', [-1.0, -0.5, 0.0])
def test_check_alpha_value(alpha, create_X_y):
    X, y = create_X_y
    with pytest.raises(ValueError):
        desmi = DESMI(alpha=alpha)
        desmi.fit(X, y)


@pytest.mark.parametrize('alpha', ['a', None, 'string', 1])
def test_check_alpha_type(alpha, create_X_y):
    X, y = create_X_y
    with pytest.raises(TypeError):
        desmi = DESMI(alpha=alpha)
        desmi.fit(X, y)


@pytest.mark.parametrize('pct_accuracy', [-1.0, -0.5, 0.0, 1.01])
def test_check_pct_accuracy_value(pct_accuracy, create_X_y):
    X, y = create_X_y
    with pytest.raises(ValueError):
        desmi = DESMI(pct_accuracy=pct_accuracy)
        desmi.fit(X, y)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method.
# In this case the test should not raise an error since this class does not
# require base classifiers that can estimate probabilities
def test_require_proba():
    X = np.random.randn(5, 5)
    y = np.array([0, 1, 0, 0, 0])
    clf1 = Perceptron()
    clf1.fit(X, y)
    DESMI([clf1, clf1, clf1])


def test_select_single_sample():
    des_mi = DESMI(pct_accuracy=0.7)
    des_mi.N_ = 2
    competences = np.array([0.7, 0.2, 1.0])
    selected_clf = des_mi.select(competences)
    expected = np.array([0, 2])
    assert np.array_equal(np.unique(selected_clf), np.unique(expected))


def test_select_batch_samples():
    n_samples = 10
    des_mi = DESMI(pct_accuracy=0.7)
    des_mi.N_ = 2
    competences = np.tile(np.array([0.7, 0.2, 1.0]), (n_samples, 1))
    selected_clf = des_mi.select(competences)
    expected = np.tile(np.array([0, 2]), (n_samples, 1))
    assert np.array_equal(np.unique(selected_clf), np.unique(expected))


def test_classify_with_ds_batch_samples():
    n_samples = 10

    # simulated predictions of the pool of classifiers
    predictions = np.tile(np.array([0, 1, 0]), (n_samples, 1))

    desmi_test = DESMI()
    desmi_test.n_classes_ = 2
    desmi_test.estimate_competence = MagicMock(
        return_value=(np.ones((n_samples, 3))))
    desmi_test.select = MagicMock(
        return_value=np.tile(np.array([[0, 2]]), (n_samples, 1)))
    result = desmi_test.classify_with_ds(predictions)
    assert np.allclose(result, np.zeros(10))


def test_predict_proba_with_ds_soft(create_pool_classifiers):
    expected = np.array([0.61, 0.39])
    DFP_mask = np.ones((1, 6))
    predictions = np.array([[0, 1, 0, 0, 1, 0]])
    probabilities = np.array([[[0.5, 0.5], [1, 0], [0.33, 0.67],
                               [0.5, 0.5], [1, 0], [0.33, 0.67]]])
    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    desmi_test = DESMI(pool_classifiers, DFP=True, voting='soft')
    desmi_test.n_classes_ = 2
    selected_indices = np.array([[0, 1, 5]])
    desmi_test.estimate_competence = MagicMock(return_value=np.ones(6))
    desmi_test.select = MagicMock(return_value=selected_indices)

    predicted_proba = desmi_test.predict_proba_with_ds(predictions,
                                                       probabilities,
                                                       DFP_mask=DFP_mask)
    assert np.isclose(predicted_proba, expected, atol=0.01).all()


def test_predict_proba_with_ds_hard(create_pool_classifiers):
    expected = np.array([0.666, 0.333])
    DFP_mask = np.ones((1, 6))
    predictions = np.array([[0, 1, 0, 0, 1, 0]])
    probabilities = np.array([[[0.5, 0.5], [1, 0], [0.33, 0.67],
                               [0.5, 0.5], [1, 0], [0.33, 0.67]]])
    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    desmi_test = DESMI(pool_classifiers, DFP=True, voting='hard')
    desmi_test.n_classes_ = 2
    selected_indices = np.array([[0, 1, 5]])
    desmi_test.estimate_competence = MagicMock(return_value=np.ones(6))
    desmi_test.select = MagicMock(return_value=selected_indices)

    predicted_proba = desmi_test.predict_proba_with_ds(predictions,
                                                       probabilities,
                                                       DFP_mask=DFP_mask)
    assert np.isclose(predicted_proba, expected, atol=0.01).all()


def test_soft_voting_no_proba(create_X_y):
    from sklearn.linear_model import Perceptron
    X, y = create_X_y
    clf = Perceptron()
    clf.fit(X, y)
    with pytest.raises(ValueError):
        DESMI([clf, clf, clf, clf], voting='soft').fit(X, y)


@pytest.mark.parametrize('voting', [None, 'product', 1])
def test_wrong_voting_value(voting, create_X_y, create_pool_classifiers):
    X, y = create_X_y
    pool = create_pool_classifiers
    with pytest.raises(ValueError):
        DESMI(pool, voting=voting).fit(X, y)
