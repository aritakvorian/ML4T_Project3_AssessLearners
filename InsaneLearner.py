import numpy as np
from BagLearner import BagLearner
from LinRegLearner import LinRegLearner
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        # create 20 BagLearners each bagging 20 LinRegLearners
        self.bags = [BagLearner(learner=LinRegLearner, kwargs={}, bags=20, boost=False, verbose=verbose) for _ in range(20)]
    def author(self):
        return "atakvorian7"
    def study_group(self):
        return "atakvorian7"
    def add_evidence(self, data_x, data_y):
        for b in self.bags:
            b.add_evidence(data_x, data_y)
    def query(self, points):
        preds = np.array([b.query(points) for b in self.bags])
        return np.mean(preds, axis=0)
