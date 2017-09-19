# sklearn_explain

This is a tool that gives model individual score explanation for an already trained scikit-learn model.

Model explanation provides the ability to interpret the effect of the predictors on the composition of an individual score. These predictors can then be ranked according to their contribution in the final score (leading to a positive or negative decision).

This is a work in progress. The set of features is evolving. Your feature requests, issues, comments, help, hints are very welcome.

# Demo
```Python
from sklearn import datasets
import pandas as pd
import sklearn_explain.explainer as expl

ds = datasets.load_breast_cancer();
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=120, random_state = 1960)

clf.fit(ds.data , ds.target)

lExplainer = expl.cModelScoreExplainer(clf)
lExplainer.fit(ds.data)
df_rc = lExplainer.explain(ds.data)

print(df_rc.head())
```

# Installation



sklearn_explain has been developed, tested and used on a python 3.5 version. 

The following commands install sklearn_explain and all its dependencies:

	pip install scipy pandas sklearn matplotlib
	pip install --upgrade git+git://github.com/antoinecarme/sklearn_explain.git
    
