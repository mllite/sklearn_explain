from sklearn import datasets
import pandas as pd
import sklearn_explain.explainer as expl

ds = datasets.load_diabetes();
from sklearn.svm import SVR
clf = SVR(max_iter=200, kernel='rbf')

clf.fit(ds.data , ds.target)

lExplainer = expl.cModelScoreExplainer(clf)
lExplainer.mSettings.mFeatureNames = ds.feature_names
lExplainer.fit(ds.data)
df_rc = lExplainer.explain(ds.data)

print(df_rc.head())
