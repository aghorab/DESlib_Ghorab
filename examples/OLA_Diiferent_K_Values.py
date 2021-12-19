# explore k in knn for DCS-LA with overall local accuracy
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from deslib.dcs.ola import OLA
from matplotlib import pyplot


# get the dataset
def get_dataset():
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
    return X, y


# get a list of models to evaluate
def get_models():
    models = dict()
    for n in range(2, 22):
        models[str(n)] = OLA(k=n)
    return models


# evaluate a give model using cross-validation
def evaluate_model(model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names, means = list(), list(), list()
for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)  # every scores object contains the result of RepeatedStratifiedKFold (10 splits, 3times)
    names.append(name)
    means.append(mean(scores))
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

print('Max Score = %.3f' % (np.max(means)))

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
