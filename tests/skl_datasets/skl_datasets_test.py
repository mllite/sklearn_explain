import numpy as np
import pandas as pd

from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.svm import *
from sklearn.neural_network import *
from sklearn.ensemble import *
from sklearn.dummy import *
from sklearn.calibration import CalibratedClassifierCV

import copy

from sklearn import datasets

import xgboost as xgb
import lightgbm as lgb

def define_tested_class_datasets():

    gDatasets = {};
    gDatasets["iris"] = datasets.load_iris()
    gDatasets["digits"] = datasets.load_digits()
    gDatasets["BreastCancer"] = datasets.load_breast_cancer();
    # gDatasets["kddcup99"] = datasets.fetch_kddcup99();
    gDatasets["BinaryClass_10"] = datasets.make_classification(n_classes=2, n_features=10, random_state=1960);
    gDatasets["FourClass_10"] = datasets.make_classification(n_classes=4, n_features=10, n_informative=4, random_state=1960);
    gDatasets["BinaryClass_100"] = datasets.make_classification(n_classes=2, n_features=100, random_state=1960);
    gDatasets["FourClass_100"] = datasets.make_classification(n_classes=4, n_samples=1000, n_informative=50, n_features=100, random_state=1960);
    gDatasets["BinaryClass_500"] = datasets.make_classification(n_classes=2, n_features=500, random_state=1960);
    gDatasets["FourClass_500"] = datasets.make_classification(n_classes=4, n_samples=1000, n_informative=100, n_features=500, random_state=1960);
    return gDatasets;


def get_human_friendly_name(model):
    # print(model.__dict__ )
    if(hasattr(model , 'kernel')):
        lkernel = model.kernel
        return str(model.__class__.__name__) + "_" + str(lkernel)
    if(hasattr(model , 'method')):
        lmethod = model.method
        return str(model.__class__.__name__) + "_" + str(lmethod)
    return str(model.__class__.__name__);



def define_tested_classifiers():
    lNbEstimatorsInEnsembles = 4
    base_classifiers = [DecisionTreeClassifier(max_depth=5, random_state = 1960) ,
                        AdaBoostClassifier(n_estimators=lNbEstimatorsInEnsembles, random_state = 1960),
                        GradientBoostingClassifier(n_estimators=lNbEstimatorsInEnsembles, random_state = 1960),
                        SGDClassifier( random_state = 1960),
                        LogisticRegression( random_state = 1960),
                        RandomForestClassifier(n_estimators=lNbEstimatorsInEnsembles, random_state = 1960),
                        GaussianNB(),
                        SVC(max_iter=200, probability=True, kernel='linear', decision_function_shape='ovr', random_state = 1960),
                        SVC(max_iter=400, probability=True, kernel='poly', decision_function_shape='ovr', random_state = 1960),
                        SVC(max_iter=200, probability=True, kernel='rbf', decision_function_shape='ovr', random_state = 1960),
                        SVC(max_iter=200, probability=True, kernel='sigmoid', decision_function_shape='ovr', random_state = 1960),
                        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 5), random_state=1960),
                        RidgeClassifier(random_state = 1960),
                        xgb.XGBClassifier(n_estimators=10, nthread=1, min_child_weight=10, max_depth=3, seed=1960),
                        BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5, random_state = 1960),
                                          random_state = 1960),
                        CalibratedClassifierCV(base_estimator=DecisionTreeClassifier(max_depth=5, random_state = 1960), cv=4, method='sigmoid'),
                        CalibratedClassifierCV(base_estimator=DecisionTreeClassifier(max_depth=5, random_state = 1960), cv=4, method='isotonic'),
                        lgb.LGBMClassifier(num_leaves=40, learning_rate=0.05, n_estimators=10)
                        ]
    
    tested_classifiers = {};
    for (i, reg) in enumerate(base_classifiers) :
        name = get_human_friendly_name(reg)
        name = name + "_" + str(i);
        tested_classifiers[name] = reg;
    # print(tested_classifiers.keys())
    return tested_classifiers;


def test_class_dataset_and_model(ds_name, model_name):
    lDatasets = define_tested_class_datasets();
    ds = lDatasets[ds_name];
    lClassifiers = define_tested_classifiers();
    model = copy.deepcopy(lClassifiers[model_name]);
    # dumpModel(model)
    lName = get_human_friendly_name(model)

    if(isinstance(ds, (list, tuple))):
        X = ds[0]
        y = ds[1]
    else:
        X = ds.data
        y = ds.target

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1960)
    # print(X_train[0:1 , ], y_train[0:1], X_test[0:1 , ], y_test[0:1])

    model.fit(X_train , y_train)

    import sklearn_explain.explainer as expl

    lExplainer = expl.cModelScoreExplainer(model)
    if(hasattr(ds , "feature_names")):
        lExplainer.mSettings.mFeatureNames= ds.feature_names
    if(hasattr(ds , "target_names")):
        lExplainer.mSettings.mClasses = ds.target_names
    lExplainer.fit(X_train)
    df_rc = lExplainer.explain(X_test)
    
    print(df_rc.shape)
    print(df_rc.columns)
    NC = df_rc.shape[1]
    print(df_rc[[col for col in df_rc.columns if col.startswith('reason_')]].describe())
    print(df_rc.sample(6, random_state=1960))


# test_class_dataset_and_model("BinaryClass_10" , "RandomForestClassifier_6")
