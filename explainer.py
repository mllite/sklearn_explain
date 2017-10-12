from sklearn_explain.reason_codes import explain_factory as expl
from sklearn_explain.reason_codes import settings as conf

# This is a model explainer designed for classifiers and regressors with continous inputs.
# Support for multiclass models and categorical inputs is also available.
# usage :

'''
clf = RandomForestClassifier().fit(X,y) # ... train any scikit-learn model 
lExplainer = cModelScoreExplainer(clf).fit(X,y) # 
reason_codes = lExplainer.explain(X_new)
'''

class cModelScoreExplainer:

    def __init__(self , clf):
        self.mModel = clf
        self.mImplementation = None
        self.mSettings = conf.cScoreExplainerConfig()
        self.mDebug = True


    def fit(self, X):
        lFactory = expl.cScoreExplainerFactory()
        self.mImplementation = lFactory.build_Explainer(self.mModel , self.mSettings)
        return self.mImplementation.fit(X)


    def explain(self, X):
        assert(self.mImplementation is not None)
        return self.mImplementation.explain(X)
    

    def get_local_score_card(self, X):
        assert(self.mImplementation is not None)
        return self.mImplementation.get_local_score_card(X)
        
