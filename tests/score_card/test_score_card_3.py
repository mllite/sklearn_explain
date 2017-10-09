from sklearn import datasets
import pandas as pd
import sklearn_explain.explainer as expl

ds = datasets.load_breast_cancer();
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=120, random_state = 1960)

NC = 2
X = ds.data[:,0:NC]
clf.fit(X , ds.target)

lExplainer = expl.cModelScoreExplainer(clf)
lExplainer.mSettings.mFeatureNames = [c for c in "ABCDEFGHIJKLMNOP"][0:NC]
lExplainer.mSettings.mScoreBins = 3
lExplainer.mSettings.mFeatureBins = 3
lExplainer.fit(X)
df_rc = lExplainer.explain(X)

print(df_rc.head())

scorecard_df = lExplainer.get_local_score_card(X[0,:].reshape(1,-1))
print(scorecard_df.head(scorecard_df.shape[0]))
