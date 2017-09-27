import pandas as pd
import numpy as np

from . import settings as conf
from . import explain_abstract as exp

class cRegressionModel_ScoreExplainer(exp.cAbstractScoreExplainer):

    def __init__(self , clf, settings = None):
        exp.cAbstractScoreExplainer.__init__(self, clf, settings);

    def get_score(self, X):
        if(hasattr(self.mClassifier , 'predict')):
            lScore = self.mClassifier.predict(X)
            return lScore
        return None

