from sklearn import datasets
import numpy as np
import pandas as pd
import sklearn_explain.explainer as expl

ds = datasets.load_iris();
print(ds.__dict__)

C1 = np.random.randint(3, size=ds.data.shape[0]).T.reshape(-1,1)
C2 = np.random.randint(5, size=ds.data.shape[0]).T.reshape(-1,1)
C2= C2 * 5

print(C1.shape, C2.shape, ds.data.shape)

X = np.concatenate((C1, C2, ds.data) , axis=1)
print(C1.shape, C2.shape, ds.data.shape , X.shape)
print(X[0:5,:])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
clf = Pipeline([
    ('feature_selection', SelectKBest(chi2, k=2)),
    ('classification', RandomForestClassifier(n_estimators=12, random_state = 1960))])
clf.fit(X , ds.target)


lExplainer = expl.cModelScoreExplainer(clf)
lExplainer.mSettings.mFeatureNames = ["C1" , "C2"] + ds.feature_names
lExplainer.mSettings.mCategoricalFeatureNames = ["C1" , "C2"]
lExplainer.mSettings.mScoreBins = 10
lExplainer.mSettings.mFeatureBins = 10
lExplainer.mSettings.mMaxReasons = 2
lExplainer.mSettings.mExplanationOrder = 2
# lExplainer.mSettings.mClasses = ds.target_names # ['setosa', 'versicolor', 'virginica']
# lExplainer.mSettings.mMainClass = 'versicolor'
lExplainer.fit(X)
df_rc = lExplainer.explain(X)

cols = [col for col in df_rc.columns if col.startswith("C")] 
print(df_rc[cols].head())

cols = [col for col in df_rc.columns if col.startswith("detailed_")] 
print(df_rc[cols].head())
