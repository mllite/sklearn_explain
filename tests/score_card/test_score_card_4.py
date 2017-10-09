from sklearn import datasets
import pandas as pd
import numpy as np
import sklearn_explain.explainer as expl

ds = datasets.load_breast_cancer();
from sklearn.linear_model import Ridge
clf = Ridge(random_state = 1960)

np.random.seed(1960);
NR = 1000
NC = 4
X = np.random.rand(NR,NC)
y = ((np.sum(X, axis=1) > 0.25) * (np.sum(X, axis=1) < 0.5))
clf.fit(X , y)

lExplainer = expl.cModelScoreExplainer(clf)
lExplainer.mSettings.mFeatureNames = [c for c in "ABCDEFGHIJKLMNOP"][0:NC]
lExplainer.mSettings.mScoreBins = 10
lExplainer.mSettings.mFeatureBins = 10
lExplainer.mSettings.mExplanationOrder = 1
lExplainer.fit(X)
df_rc = lExplainer.explain(X)

print(df_rc.head())

scorecard_df = lExplainer.get_local_score_card(X[0,:].reshape(1,-1))
print(scorecard_df.head(scorecard_df.shape[0]))
