from sklearn import datasets
import pandas as pd
import sklearn_explain.explainer as expl


# %matplotlib inline

ds = datasets.load_breast_cancer();
NC = 4
lFeatures = ds.feature_names[0:NC]

df_orig = pd.DataFrame(ds.data[:,0:NC] , columns=lFeatures)
df_orig['TGT'] = ds.target
df_orig.sample(6, random_state=1960)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=120, random_state = 1960)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_orig[lFeatures].values, 
                                                    df_orig['TGT'].values, 
                                                    test_size=0.2, 
                                                    random_state=1960)

df_train = pd.DataFrame(X_train , columns=lFeatures)
df_train['TGT'] = y_train
df_test = pd.DataFrame(X_test , columns=lFeatures)
df_test['TGT'] = y_test

clf.fit(X_train , y_train)

###########################################

lExplainer = expl.cModelScoreExplainer(clf)
lExplainer.fit(X_train)
df_rc = lExplainer.explain(X_test)

print(df_rc.head())
print(df_rc[['reason_' + str(NC-c) for c in range(NC)]].describe())

