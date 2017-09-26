import pandas as pd
import numpy as np

from . import settings as conf
from . import explain_classification as exp_class

class cScoreExplainerFactory:

    def __init__(self):
        pass

    def build_Explainer(self, clf, settings):
        if(hasattr(clf , 'predict_proba')):
            return exp_class.cClassificationModel_ScoreExplainer(clf , settings)
        if(hasattr(clf , 'decision_function')):
            return exp_class.cClassificationModel_ScoreExplainer(clf , settings)
        return None

