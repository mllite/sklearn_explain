from sklearn import datasets
import pandas as pd
import sklearn_explain.explainer as expl

ds = datasets.load_iris();
print(ds.__dict__)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=12, random_state = 1960)

clf.fit(ds.data , ds.target)

lExplainer = expl.cModelScoreExplainer(clf)
lExplainer.mSettings.mFeatureNames = ds.feature_names
lExplainer.mSettings.mScoreBins = 10
lExplainer.mSettings.mFeatureBins = 10
lExplainer.mSettings.mMaxReasons = 2
lExplainer.mSettings.mExplanationOrder = 1
lExplainer.mSettings.mClasses = ds.target_names # ['setosa', 'versicolor', 'virginica']
lExplainer.mSettings.mMainClass = 'virginica'
lExplainer.fit(ds.data)
df_rc = lExplainer.explain(ds.data)

print(df_rc.head())
