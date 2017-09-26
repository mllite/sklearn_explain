import pandas as pd
import numpy as np

class cScoreExplainerConfig:

    def __init__(self):
        self.mFeatureNames = None
        self.mScoreBins = 5 # score binning
        self.mFeatureBins = 5 # feature binning
        self.mMaxReasons = 5 # max number of explanations
        self.mExplanationOrder = 2 # max number of feature by explanation
        self.mDebug = True


