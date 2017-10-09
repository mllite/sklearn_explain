import pandas as pd
import numpy as np

from . import settings as conf
from . import explain_abstract as exp

class cClassificationModel_ScoreExplainer(exp.cAbstractScoreExplainer):

    def __init__(self , clf, settings = None):
        exp.cAbstractScoreExplainer.__init__(self, clf, settings);
        self.report_score_type()

    def report_score_type(self):
        if(hasattr(self.mClassifier , 'predict_proba')):
            if(self.mSettings.mDebug):
                print("USING_LOG_ODDS_AS_SCORE")
        elif(hasattr(self.mClassifier , 'decision_function')):
            if(self.mSettings.mDebug):
                print("USING_DECISION_FUNCTION_AS_SCORE")
        pass

    def get_score(self, X):
        class_idx = self.mSettings.get_class_index()
        if(hasattr(self.mClassifier , 'predict_proba')):
            lProba = self.mClassifier.predict_proba(X)[:,class_idx]
            lProba = lProba.clip(0.00001 , 0.99999)
            lOdds = lProba / (1.0 - lProba)
            return np.log(lOdds)
        if(hasattr(self.mClassifier , 'decision_function')):
            lDecision = self.mClassifier.decision_function(X)
            if(len(lDecision.shape) == 1):
                # binary classifier : RidgeClassifier and SGDClassifier
                return lDecision
            return lDecision[:,class_idx]
        return None

