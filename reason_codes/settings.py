import pandas as pd
import numpy as np

class cScoreExplainerConfig:

    def __init__(self):
        self.mFeatureNames = None
        self.mCategoricalFeatureNames = None
        self.mScoreBins = 5 # score binning
        self.mFeatureBins = 5 # feature binning
        self.mCustomFeatureQuantiles = None
        self.mCustomScoreQuantiles = None
        self.mMaxReasons = 5 # max number of explanations
        self.mExplanationOrder = 2 # max number of feature by explanation
        self.mClasses = None
        self.mMainClass = None
        self.mDebug = True

    def get_class_index(self):
        default_class_idx = -1
        if(self.mMainClass is None):
            return default_class_idx
        if(self.mClasses is None):
            return default_class_idx
        return list(self.mClasses).index(self.mMainClass)
    


    def is_categorical(self, feature_name):
        if(self.mCategoricalFeatureNames is not None and feature_name in self.mCategoricalFeatureNames):
            return True
        return False
