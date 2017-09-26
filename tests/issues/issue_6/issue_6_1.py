from sklearn import datasets
import pandas as pd
import sklearn_explain.explainer as expl

ds = datasets.load_breast_cancer();
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=12, random_state = 1960)

clf.fit(ds.data , ds.target)

lExplainer = expl.cModelScoreExplainer(clf)
lExplainer.mSettings.mFeatureNames = ds.feature_names
lExplainer.mSettings.mScoreBins = 10
lExplainer.mSettings.mFeatureBins = 10
lExplainer.mSettings.mMaxReasons = 2
lExplainer.mSettings.mExplanationOrder = 1
lExplainer.fit(ds.data)
df_rc = lExplainer.explain(ds.data)

print(df_rc.head())
