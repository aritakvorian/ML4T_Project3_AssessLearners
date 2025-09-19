import numpy as np
import random

class BagLearner(object):
    def __init__(self, learner, kwargs=None, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = {} if kwargs is None else kwargs
        self.bags = int(bags)
        self.boost = boost
        self.verbose = verbose
        self.learners = []

    def author(self):
        return "atakvorian7"

    def study_group(self):
        return "atakvorian7"

    def add_evidence(self, data_x, data_y):
        X = np.array(data_x)
        Y = np.array(data_y).reshape(-1)
        n = X.shape[0]
        self.learners = []
        for i in range(self.bags):
            idxs = np.random.randint(0, n, n)  # bootstrap with replacement
            Xs = X[idxs]
            Ys = Y[idxs]
            inst = self.learner(**self.kwargs)
            inst.add_evidence(Xs, Ys)
            self.learners.append(inst)

    def query(self, points):
        pts = np.array(points)
        if not self.learners:
            raise ValueError("BagLearner has not been trained.")
        preds = np.array([learner.query(pts) for learner in self.learners])
        # average across learners
        return np.mean(preds, axis=0)
