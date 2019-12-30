# -*- coding: utf-8 -*-
"""
This class implements the RUSBoost Algorithm. For more details on the theoretical description of the algorithm please
refer to the following paper: C. Seiffert, T.M. Khoshgoftaar, J. Van Hulse and A. Napolitano, "RUSBoost: A Hybrid
Approach to Alleviating Class Imbalance, IEEE Transaction on Systems, Man and Cybernetics-Part A: Systems and Human,
Vol.40(1), January 2010.
Methods:
    0- __init__(base_estimator=None, n_estimators=10, learning_rate=1.0):
            base_estimator = string to choose an algorithm. Current choice is 'tree'.
            n_estimators = Number of boosting iterations
            learning_rate = The learning rate is how quickly a network abandons old beliefs for new ones
    1- fit(X, y, base_estimator, n_estimators, learning_rate):
       Inputs:
            X = Predictors for training data as matrix/DataFrame/Dict
            y = Response (target) data as matrix
        Output: EnsembleRUSBoosting object
    2- predict(X):
       Inputs:
            X = Predictors for training data as matrix/DataFrame/Dict
       Outputs:
            Labels = predicted class labels
            scores = scores for classification ensemble

__author__ = 'Mohammed Alfaki <mohammed.alfaki@ge.com>'

First created on 2017-Aug-29, London

"""

from __future__ import division
import numpy as np
import pandas as pd
from math import log
from sklearn import tree

import lime.lime_tabular as lmt
from sklearn.externals import joblib
from sklearn.utils import check_array
from treeinterpreter import treeinterpreter
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, StandardScaler
from ModelUtils import classCount, log_writer, ismember


class EnsembleRUSBoosting(object):
    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0, with_replacement=None, random_state=None):
        self.X = []
        self.y = []
        self.W = []
        self.prior = []
        self.fit_data = []
        self.classes = []
        self.estimators = []
        self.combiner = []
        self.is_cached = []
        self.n_features = None
        self.estimator_weights = []
        self.feature_importances_ = []
        self.with_replacement = with_replacement
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.class_names = []
        self.base_estimator = base_estimator
        self.cached_score = []
        self.cached_contributions = []
        self.response = []
        self.defaultScore = -np.inf
        self.features = []
        self.ratio_to_smallest = np.array([1, 1])
        self.last_use_obs_for_iter = []
        self.nonzero_prob_classes = np.array([0, 1])
        self.random_state = random_state
        self.scaler = type('', (), {})()

    def fit(self, X, y):
        num_rows, num_cols = X.shape
        self.n_features = num_cols

        if isinstance(X, pd.DataFrame):
            self.features = X.columns.values
        else:
            self.features = np.array(['feature_%d' % i for i in range(num_cols)])

        if isinstance(y, pd.DataFrame):
            self.response = y.columns.values
        else:
            self.response = 'target'

        if self.base_estimator is None:
            self.base_estimator = 'tree'

        X = check_array(X, accept_sparse="csc")
        y = check_array(y, ensure_2d=False, dtype=None)
        class_names = np.unique(y)
        num_classes = len(class_names)

        classes = classCount(class_names, y)
        W = 1.0 / num_rows * np.ones(num_rows)
        fit_data = np.tile(np.reshape(W, (-1, 1)), (1, num_classes))
        fit_data *= np.logical_not(classes)
        if np.any(fit_data):
            fit_data /= fit_data.sum()
        else:
            fit_data = np.zeros((num_rows, num_classes))

        self.X = X
        self.y = y
        self.W = W
        self.fit_data = fit_data
        self.prior = classes.sum(axis=0) / len(y)
        self.classes = classes
        self.class_names = class_names

        hypotheses = []
        pseudo_loss = np.full(self.n_estimators, None)
        # Boosting iterations
        t = 0
        while t < self.n_estimators:
            X_s, y_s, W_s, fitData_s = self.random_under_sampler()

            # Training a weak learner. 'score' is the weak hypothesis. However, the hypothesis function is encoded in
            # 'hypotheses'
            if self.base_estimator == 'svm':
                model = []
            elif self.base_estimator == 'tree':
                model = tree.DecisionTreeClassifier(min_samples_leaf=5)
                model.fit(X_s, y_s, sample_weight=W_s)
            elif self.base_estimator == 'knn':
                model = KNeighborsClassifier(n_neighbors=3)
                model.fit(X_s, y_s)
            elif self.base_estimator == 'logistic':
                model = LogisticRegression(random_state=self.random_state)
                model.fit(X_s, y_s, sample_weight=W_s)
            elif self.base_estimator == 'nn':
                model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10),
                                      random_state=self.random_state)
                model.fit(X_s, y_s)
            else:
                model = []

            score = model.predict_proba(self.X)
            _, pos = ismember(model.classes_, self.class_names)
            s = np.zeros((num_rows, num_classes))
            s[:, pos] = score

            # Get margin per class, that is, score for the true class minus score for this class. For the true class,
            # the margin is the score for the true class
            margin = np.tile(np.reshape((self.classes * s).sum(axis=1), (-1, 1)), (1, 2)) - np.logical_not(self.classes) * s

            # fit_data is matrix of size NxK with weights for false labels. For true labels, weights are zero. Make
            # sure it is properly normalized
            falseWperObs = self.fit_data.sum(axis=1)
            useObs = (self.W > 0) & (falseWperObs > 0)
            if np.logical_not(np.any(useObs)):
                return

            self.fit_data[useObs, :] = self.fit_data[useObs, :] * np.tile(
                np.reshape(self.W[useObs] / self.W[useObs].sum() / falseWperObs[useObs], (-1, 1)), (1, 2))
            Wtot = self.W.sum()
            self.fit_data[np.logical_not(useObs), :] = 0

            # Get pseudo-loss
            loss = 0.5 * (self.fit_data * (1 - margin)).sum()

            # Update weights
            beta = (loss / (1 - loss)) ** self.learning_rate
            self.fit_data *= beta ** ((1 + margin) / 2)

            # Get new weights for data generation. Sum of weights remains constant.
            Wnew = self.fit_data.sum(axis=1)
            self.W = Wnew * Wtot / Wnew.sum()

            hypotheses.append(model)  # Hypothesis function
            pseudo_loss[t] = loss  # Pseudo-loss at each iteration

            # Incrementing loop counter
            t += 1

            # Log message
            if (t % 100) == 0:
                log_writer('Boosting iteration #%d' % t)

        pseudo_loss[(pseudo_loss > 0.5)] = 0.5
        estimator_weights = np.full(self.n_estimators, np.nan)
        for i in range(self.n_estimators):
            estimator_weights[i] = 0.5 * self.learning_rate * log((1 - pseudo_loss[i]) / pseudo_loss[i])

        self.estimators = hypotheses
        self.estimator_weights = estimator_weights
        self.get_feature_importances()

        self.scaler = StandardScaler(with_mean=False)
        self.scaler.fit(self.X)

        self.X = []
        self.y = []
        self.W = []
        self.fit_data = []
        self.classes = []
        self.is_cached = []
        self.cached_score = []
        self.last_use_obs_for_iter = []

    def random_under_sampler(self, with_replacement=None):
        num_classes = len(self.class_names)
        NumPerClass = self.classes.sum(axis=0)
        NumToSample = np.ceil(min(NumPerClass) * self.ratio_to_smallest).astype(int)
        NumToSample[(NumPerClass == 0)] = 0

        # Loop over classes and sample the desired number of observations for each class
        idx = np.zeros(NumToSample.sum(), dtype=int)
        idxbegin = 0
        for k in range(num_classes):
            if NumToSample[k] > 0:
                idxk = np.where(self.classes[:, k])[0]
                if NumPerClass[k] != NumToSample[k]:
                    if NumPerClass[k] < NumToSample[k]:
                        replaceArgs = True
                    else:
                        replaceArgs = False

                    if self.with_replacement is not None:
                        replaceArgs = self.with_replacement

                    prob = self.W[idxk] / self.W[idxk].sum()
                    idxk = np.random.choice(idxk, NumToSample[k], p=prob, replace=replaceArgs)

                idx[range(idxbegin, idxbegin + NumToSample[k])] = idxk
                idxbegin = idxbegin + NumToSample[k]

        # Shuffle in case the weak learner is sensitive to the order in which observations are found
        idx = idx[np.random.permutation(len(idx))]
        self.last_use_obs_for_iter = idx

        # Return
        X = self.X[idx, :]
        y = self.y[idx]
        W = np.ones(len(idx))  # weights have been used for sampling
        fitData = self.fit_data[idx, :]

        return X, y, W, fitData

    def complete_prediction(self, X):
        # Initialize
        self.cached_score = []
        self.cached_contributions = []
        self.is_cached = np.full(self.n_estimators, False)

        # Get sizes
        X = check_array(X, accept_sparse="csc")
        num_rows, _ = X.shape

        # Predict. The combiner object aggregates scores from individual learners. The score returned by combiner is the
        # aggregated score.
        scores = np.tile(self.defaultScore, (num_rows, max(1, len(self.class_names))))
        cls = np.argmax(self.prior)
        labels = np.tile(self.class_names[cls], num_rows)
        if self.n_estimators == 0:
            return

        for t in range(self.n_estimators):
            weak = self.estimators[t]
            scores = weak.predict_proba(X)
            scores = self.update_cache(scores, t)

        # Transform scores and find the most probable class
        not_nan = np.logical_not(np.all(np.isnan(scores) | (scores == self.defaultScore), axis=1))
        class_num = np.argmax(scores[not_nan, :], axis=1)
        labels[not_nan] = self.class_names[class_num]

        return labels, scores

    def compact_model(self):
        self.X = []
        self.y = []
        self.W = []
        self.fit_data = []
        self.classes = []
        self.is_cached = []
        self.cached_score = []
        self.last_use_obs_for_iter = []

    def predict(self, X):
        # Get labels and scores from predict
        labels, _ = self.complete_prediction(X)
        return labels

    def predict_proba(self, X):
        # Get labels and scores from predict
        _, scores = self.complete_prediction(X)

        # Normalize scores
        scores = normalize(scores, axis=1, norm='l1')

        return scores

    def get_feature_importances(self):
        importances = np.full(self.n_features, 0.0)
        for t in range(self.n_estimators):
            if not hasattr(self.estimators[t], 'feature_importances_'):
                importances = np.full(self.n_features, 1.0)
                break

            importances = importances + self.estimator_weights[t] * self.estimators[t].feature_importances_

        self.feature_importances_ = importances / importances.sum()

    def update_cache(self, scores, t):
        # If first caching operation, make CachedScore. Else reset combined values to zeros.
        if len(self.cached_score) == 0:
            self.cached_score = np.full(scores.shape, 0)
        else:
            tf = np.isnan(self.cached_score)
            if np.any(tf):
                self.cached_score[tf] = 0

        # If the learner has been already combined or has non-positive weight, do nothing
        if self.is_cached[t] and (self.estimator_weights[t] <= 0):
            return

        # Add
        self.cached_score = self.cached_score + scores * self.estimator_weights[t]
        self.is_cached[t] = True
        scores = self.cached_score

        # score
        tf, _ = ismember(self.class_names, self.nonzero_prob_classes)
        scores[:, np.logical_not(tf)] = self.defaultScore

        return scores

    def predict_contributions_treeinterp(self, X):
        # Initialize
        self.cached_contributions = []
        self.is_cached = np.full(self.n_estimators, False)

        # Get sizes
        X = check_array(X, accept_sparse="csc")
        num_rows, num_cols = X.shape

        contributions = None
        for t in range(self.n_estimators):
            weak = self.estimators[t]
            _, _, contributions = treeinterpreter.predict(weak, X)
            contributions = self.update_cache_contributions(contributions, t)

        contr_matrix, feature_ordering = np.full(X.shape, 0.0), np.full(X.shape, 0)
        for i in range(num_rows):
            sorted_contrib = sorted(zip(contributions[i, :, 1], range(num_cols)), key=lambda x: -abs(x[0]))
            contr_matrix[i, :] = map(abs, zip(*sorted_contrib)[0])
            feature_ordering[i, :] = zip(*sorted_contrib)[1]

        contr_matrix = normalize(contr_matrix, axis=1, norm='l1')
        return contr_matrix, feature_ordering

    def predict_contributions_lime(self, X):
        # Get sizes
        X = check_array(X, accept_sparse="csc")
        num_rows, num_cols = X.shape

        explainer = lmt.LimeTabularExplainer(X[:1], feature_names=self.features, class_names=self.class_names, discretize_continuous=False,
                                             random_state=self.random_state)
        explainer.scaler = self.scaler

        contr_matrix, feature_ordering = np.full(X.shape, 0.0), np.full(X.shape, 0)
        for i in range(num_rows):
            exp = explainer.explain_instance(X[i, :], self.predict_proba, num_features=self.n_features)
            sorted_contrib = exp.as_map()[1]
            contr_matrix[i, :] = map(abs, zip(*sorted_contrib)[1])
            feature_ordering[i, :] = zip(*sorted_contrib)[0]
            del exp

        contr_matrix = normalize(contr_matrix, axis=1, norm='l1')
        return contr_matrix, feature_ordering

    def update_cache_contributions(self, contributions, t):
        # If first caching operation, make CachedScore. Else reset combined values to zeros.
        if len(self.cached_contributions) == 0:
            self.cached_contributions = np.full(contributions.shape, 0)
        else:
            tf = np.isnan(self.cached_contributions)
            if np.any(tf):
                self.cached_contributions[tf] = 0

        # Add
        self.cached_contributions = self.cached_contributions + contributions * self.estimator_weights[t]
        self.is_cached[t] = True
        contributions = self.cached_contributions

        return contributions


if __name__ == '__main__':
    ecoli_dataset_file = 'data/ecoli_dataset.csv'
    data = pd.read_csv(ecoli_dataset_file)
    predictors = ['var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6']
    response = 'target'
    model = EnsembleRUSBoosting(base_estimator='tree', n_estimators=1500, learning_rate=0.1, with_replacement=False, random_state=0)
    model.fit(data[predictors], data[response])
    joblib.dump(model, 'test.pkl')

    loaded_model = joblib.load('test.pkl')
    labels, scores = loaded_model.complete_prediction(data[predictors])
    # contr_matrix_i, feature_ordering_i = loaded_model.predict_contributions_treeinterp(data[predictors])
    # contr_matrix_l, feature_ordering_l = loaded_model.predict_contributions_lime(data[predictors])

    data['ypred'] = labels
    data['score_False'] = scores[:, 0]
    data['score_True'] = scores[:, 1]
    model_output_file = 'data/ecoli_modeloutput_python.csv'
    data.to_csv(model_output_file, index=False)
