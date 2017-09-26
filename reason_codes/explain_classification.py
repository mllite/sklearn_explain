import pandas as pd
import numpy as np

from . import settings as conf
from . import explain_abstract as exp

class cClassificationModel_ScoreExplainer(exp.cAbstractScoreExplainer):

    def __init__(self , clf, settings = None):
        exp.cAbstractScoreExplainer.__init__(self, clf, settings);

    def get_score(self, X):
        if(hasattr(self.mClassifier , 'predict_proba')):
            if(self.mSettings.mDebug):
                print("USING_PROBABILITY_AS_SCORE")
            lProba = self.mClassifier.predict_proba(X)[:,1]
            lProba = lProba.clip(0.00001 , 0.99999)
            lOdds = lProba / (1.0 - lProba)
            return np.log(lOdds)
        if(hasattr(self.mClassifier , 'decision_function')):
            if(self.mSettings.mDebug):
                print("USING_DECISION_FUNCTION_AS_SCORE")
            lDecision = self.mClassifier.decision_function(X)
            if(len(lDecision.shape) == 1):
                # binary classifier : RidgeClassifier and SGDClassifier
                return lDecision
            return lDecision[:,0]
        return None

