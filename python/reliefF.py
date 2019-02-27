from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import euclidean

# -*- coding: utf-8 -*-


class ReliefF(object):
    def __init__(self, n_features_to_keep=10):
        self.feature_scores = None
        self.top_features = None
        self.n_features_to_keep = n_features_to_keep

    def _find_nm(self, sample, X):
        """Find the near-miss of sample

        Parameters
        ----------
        sample: array-like {1, n_features}
            queried sample
        X: array-like {n_samples, n_features}
            The subclass which the label is diff from sample
        Returns
        -------
        idx: int
            index of near-miss in X

        """
        dist = 100000
        idx = None
        for i, s in enumerate(X):
            tmp = euclidean(sample, s)
            if tmp <= dist:
                dist = tmp
                idx = i

        if dist == 100000:
            raise ValueError

        return idx

    def fit(self, X, y, scaled=True):
        """Computes the feature importance scores from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        scaled: Boolen
            whether scale X ro not
        Returns
        -------
        self.top_features
        self.feature_scores
        """
        if scaled:
            X = minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)

        self.feature_scores = np.zeros(X.shape[1], dtype=np.float64)

        # The number of labels and its corresponding prior probability
        labels, counts = np.unique(y, return_counts=True)
        Prob = counts / float(len(y))

        for label in labels:
            # Find the near-hit for each sample in the subset with label 'label'
            select = (y == label)

            print(select)

            tree = KDTree(X[select, :])
            nh = tree.query(X[select, :], k=2, return_distance=False)[:, 1:]
            nh = (nh.T[0]).tolist()
            # print(nh)

            # calculate -diff(x, x_nh) for each feature of each sample
            # in the subset with label 'label'
            nh_mat = np.square(np.subtract(X[select, :], X[select, :][nh, :])) * -1

            # Find the near-miss for each sample in the other subset
            nm_mat = np.zeros_like(X[select, :])
            for prob, other_label in zip(Prob[labels != label], labels[labels != label]):

                print(prob)

                other_select = (y == other_label)
                nm = []
                for sample in X[select, :]:

                    print(sample)

                    nm.append(self._find_nm(sample, X[other_select, :]))

                # print(nm)
                # calculate -diff(x, x_nm) for each feature of each sample in the subset
                # with label 'other_label'
                nm_tmp = np.square(np.subtract(X[select, :], X[other_select, :][nm, :])) * prob
                nm_mat = np.add(nm_mat, nm_tmp)

            mat = np.add(nh_mat, nm_mat)
            self.feature_scores += np.sum(mat, axis=0)
        # print(self.feature_scores)

        # Compute indices of top features, cast scores to floating point.
        self.top_features = np.argsort(self.feature_scores)[::-1]
        self.feature_scores = self.feature_scores[self.top_features]

        return self.top_features, self.feature_scores

    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        return X[:, self.top_features[:self.n_features_to_keep]]

    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        self.fit(X, y)
        return self.transform(X)



if __name__=='__main__':

    traindata = np.array(pd.read_csv('D:/data/database/opsmile1580/zcl/train.csv'))
    data=traindata[:,0:1580]
    labels=traindata[:,1580:1581]
    value=[]
    for i in labels:
        value.append(int(i))
    label=np.array(value)
    label=np.reshape(label,(13199,))
    # # datass=np.array([[1,2,3,4,4,4,4,4,4,4,4],[1,2,4,3,2,3,1,2,3,4,2],[2,3,6,7,1,8,9,3,3,2,1],[3,2,4,3,2,1,2,3,4,4,5]])
    # # labelss=np.array([0,0,1,1])
    # # print(datass.shape)
    # # print(labelss.shape)
    # # labelsss=np.reshape(labelss,(4,1))
    # # print(labelsss.shape)
    fs=ReliefF(n_features_to_keep=10)
    top_features, feature_scores=fs.fit(data,label)
    print('topfeature:')
    print(top_features)
    print('featurescors:')
    print(feature_scores)
    top_features=pd.DataFrame(top_features)
    top_features.to_csv('D:/data/database/relieff/opsm1580/top_feature.csv',index=False)
    feature_scores = pd.DataFrame(feature_scores)
    feature_scores.to_csv('D:/data/database/relieff/opsm1580/feature_scores.csv', index=False)
    data_trans=fs.transform(data)
    print(data_trans)
    data_trans = pd.DataFrame(data_trans)
    data_trans.to_csv('D:/data/database/relieff/opsm1580/data_trans.csv', index=False)




