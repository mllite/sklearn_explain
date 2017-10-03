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


def define_tested_reg_datasets():

    gDatasets = {};
    gDatasets["diabetes"] = datasets.load_diabetes()
    gDatasets["boston"] = datasets.load_boston()
    gDatasets["freidman1"] = datasets.make_friedman1(random_state=1960)
    gDatasets["freidman2"] = datasets.make_friedman2(random_state=1960)
    gDatasets["freidman3"] = datasets.make_friedman3(random_state=1960)
    gDatasets["RandomReg_10"] = datasets.make_regression(n_features=10, random_state=1960);
    gDatasets["RandomReg_100"] = datasets.make_regression(n_features=100, random_state=1960);
    gDatasets["RandomReg_500"] = datasets.make_regression(n_features=500, random_state=1960);

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



def define_tested_regressors():
    lNbEstimatorsInEnsembles = 4
    base_regressors = [DecisionTreeRegressor(max_depth=5, random_state = 1960) ,
                       DummyRegressor(),
                       AdaBoostRegressor(n_estimators=lNbEstimatorsInEnsembles, random_state = 1960),
                       GradientBoostingRegressor(n_estimators=lNbEstimatorsInEnsembles, random_state = 1960),
                       SGDRegressor(random_state = 1960),
                       RandomForestRegressor(n_estimators=lNbEstimatorsInEnsembles, random_state = 1960),
                       Ridge(random_state = 1960),
                       SVR(max_iter=200, kernel='linear'),
                       SVR(max_iter=400, kernel='poly'),
                       SVR(max_iter=200, kernel='rbf'),
                       MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 5), random_state=1960),
                       xgb.XGBRegressor(n_estimators=10,  nthread=1, min_child_weight=10, max_depth=3, seed=1960),
                       SVR(max_iter=200, kernel='sigmoid'),
                       BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=5, random_state = 1960),
                                         random_state = 1960),
                       DecisionTreeRegressor(random_state = 1960) ,
                       lgb.LGBMRegressor(objective='regression',num_leaves=20, learning_rate=0.05, n_estimators=10)
                       ]
    
    tested_regressors = {};
    for (i, reg) in enumerate(base_regressors) :
        name = get_human_friendly_name(reg)
        name = name + "_" + str(i);
        tested_regressors[name] = reg;
    return tested_regressors;  


def test_reg_dataset_and_model(ds_name, model_name):
    lDatasets = define_tested_reg_datasets();
    ds = lDatasets[ds_name];
    lRegressors = define_tested_regressors();
    model = copy.deepcopy(lRegressors[model_name]);
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
    #lExplainer.mSettings.mExplanationOrder = 1
    lExplainer.fit(X_train)
    df_rc = lExplainer.explain(X_test)
    
    print(df_rc.shape)
    print(df_rc.columns)
    NC = df_rc.shape[1]
    print(df_rc[[col for col in df_rc.columns if col.startswith('reason_')]].describe())
    print(df_rc.sample(6, random_state=1960))

    return lExplainer
