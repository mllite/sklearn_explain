from sklearn import datasets
import pandas as pd
import numpy as np



ds = datasets.load_breast_cancer();
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_leaf=30, random_state = 1960)
NC = 12

X = ds.data[:,0:NC]
y = ds.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1960)

clf.fit(X_train , y_train)


import sklearn_explain.explainer as expl
lExplainer = expl.cModelScoreExplainer(clf)
lExplainer.mSettings.mFeatureNames = ds.feature_names[0:NC]
lExplainer.mSettings.mExplanationOrder = 1
    
lExplainer.fit(X_train)
df_rc = lExplainer.explain(X_test)

print(df_rc.columns)

df_rc_1 = lExplainer.explain(X_test[0].reshape(1, -1))
print(df_rc_1[[col for col in df_rc_1.columns if col.startswith('detailed')]])

# Explain the score = ln(p(1) / (1 - p(1)))

lFeature_Quantiles = {
'mean area': {0: -np.inf,
  1: 571.85},
'mean concave points': {0: -np.inf,
  1: 0.51},
'mean perimeter': {0: -np.inf,
  1: 98.31},
'radius error': {0: -np.inf,
  1: 0.354}
}

lExplainer2 = expl.cModelScoreExplainer(clf)
lExplainer2.mSettings.mFeatureNames = ds.feature_names[0:NC]
lExplainer2.mSettings.mCustomFeatureQuantiles = lFeature_Quantiles
lExplainer2.mSettings.mExplanationOrder = 1
    
lExplainer2.fit(X_train)
df_rc2 = lExplainer2.explain(X_test)

print(df_rc2.columns)

df_rc_2 = lExplainer2.explain(X_test[0].reshape(1, -1))
print(df_rc_2[[col for col in df_rc_2.columns if col.startswith('detailed')]])
