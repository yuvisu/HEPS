import numpy as np
import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.neural_network import MLPClassifier

class FeatureShapePrinter(BaseEstimator, TransformerMixin):
    def __init__(self, message=""):
        self.message = message
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print(f"{self.message} shape: {X.shape}")
        return X

def model_construction(pm_config, X_val = None, y_val= None, n_jobs = 1):
    
    model_alg = pm_config.model_alg.split("_")[0]+"_"
    
    if model_alg == "xgboost_":
        return xgb_construction(pm_config, X_val, y_val, n_jobs)
    elif model_alg == "lrc_":
        return lrc_construction(pm_config, X_val, y_val, n_jobs)
    elif model_alg == "rfc_":
        return rfc_construction(pm_config, X_val, y_val, n_jobs)
    elif model_alg == "mlp_":
        return mlp_construction(pm_config, X_val, y_val, n_jobs)
    else:
        return None

def model_finalize(pm_config, X_val = None, y_val= None):
    
    model_alg = pm_config.model_alg.split("_")[0]+"_"
    
    if model_alg == "xgboost_":
        return xgb_finalize(pm_config, X_val, y_val)
    elif model_alg == "lrc_":
        return lrc_finalize(pm_config, X_val, y_val)
    elif model_alg == "rfc_":
        return rfc_finalize(pm_config, X_val, y_val)
    elif model_alg == "mlp_":
        return mlp_finalize(pm_config, X_val, y_val)
    else:
        return None

def model_fml(pm_config, X_val = None, y_val= None):
    
    model_alg = pm_config.model_alg.split("_")[0]+"_"
    
    if model_alg == "xgboost_":
        return fml_xgb(pm_config, X_val, y_val)
    elif model_alg == "lrc_":
        return fml_lrc(pm_config, X_val, y_val)
    else:
        return None
    
def prep_construction():
    #imputation
    simim = SimpleImputer()
    #preprocessing
    scaler = MinMaxScaler()
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
    ])
    
    return pipe

def imb_construction(pm_config):
    
    if pm_config.setting_params.imb_type == "ros":
        ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    elif pm_config.setting_params.imb_type == "rus":
        ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    else:
        ibp = None
    
    return ibp

def fml_xgb(pm_config, X_val = None, y_val= None):
    model_name = "fml_xgb"
    
    base_model = xgb.XGBClassifier(verbosity = pm_config.setting_params.verbosity, 
                                   random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs,
                                   max_depth = pm_config.max_depth,
                                   min_child_weight = pm_config.min_child_weight,
                                   gamma = pm_config.gamma,
                                   subsample = pm_config.subsample,
                                   colsample_bytree = pm_config.colsample_bytree,
                                   eta = pm_config.eta,
                                   n_estimators = pm_config.n_estimators,
                                   reg_alpha = pm_config.reg_alpha,
                                   reg_lambda = pm_config.reg_lambda)
    
    fit_params={
        "eval_set": [[X_val, y_val]],
        "eval_metric": pm_config.setting_params.eval_metric,
        "early_stopping_rounds": pm_config.setting_params.early_stopping_rounds}
    
    return base_model, model_name, fit_params

def fml_lrc(pm_config, X_val = None, y_val= None):
    model_name = "fml_lrc"
    
    base_model = LogisticRegression(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs,
                                   C = pm_config.C,
                                   solver = pm_config.solver,
                                   penalty = pm_config.penalty)
    
    fit_params={}
    
    return base_model, model_name, fit_params
    
def xgb_construction(pm_config, X_val = None, y_val= None, n_jobs = 1):

    print('* using xgb')
    model_name="xgboost_"
    
    #imputation
    simim = SimpleImputer(strategy="mean")
    
    #preprocessing
    scaler = MinMaxScaler() 
    
    #imbalance processing
    if pm_config.setting_params.imb_type == "ros":
        ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    elif pm_config.setting_params.imb_type == "rus":
        ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    else:
        ibp = None
        
    base_model = xgb.XGBClassifier(verbosity = None, random_state=pm_config.setting_params.random_state,
        eval_metric=pm_config.setting_params.eval_metric,
        early_stopping_rounds=pm_config.setting_params.early_stopping_rounds)

    hyper_parameters = {
                        "base__max_depth": pm_config.range_params.max_depth,
                        "base__min_child_weight":pm_config.range_params.min_child_weight,
                        "base__gamma": pm_config.range_params.gamma,
                        'base__subsample':pm_config.range_params.subsample,
                        'base__colsample_bytree':pm_config.range_params.colsample_bytree,
                        "base__eta" : pm_config.range_params.eta,
                        "base__n_estimators": pm_config.range_params.n_estimators,
                        "base__reg_alpha": pm_config.range_params.reg_alpha,
                        "base__reg_lambda": pm_config.range_params.reg_lambda
                       }
    
    fit_params={
        # "base__eval_metric": pm_config.setting_params.eval_metric,
        # "base__early_stopping_rounds": pm_config.setting_params.early_stopping_rounds,
        "base__verbose": False,
        "base__eval_set": [[X_val, y_val]]}
    
    if ibp is None:
        pipe = Pipeline([
            ("imputation", simim),
            ("scaler", scaler),
            ("base", base_model)
        ])
    else:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("imbalance",ibp),
            ("base", base_model)
        ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, scoring="f1", verbose=pm_config.setting_params.cv_verbose, cv = pm_config.setting_params.cv_fold, n_jobs=n_jobs, error_score='raise')
    return clf, model_name, fit_params

def xgb_finalize(pm_config, X_val = None, y_val= None):
    model_name="xgb_FINAL_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    
    #imbalance processing
    if pm_config.setting_params.imb_type == "ros":
        ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    elif pm_config.setting_params.imb_type == "rus":
        ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    else:
        ibp = None
        
    #base model
    base_model = xgb.XGBClassifier(verbosity = pm_config.setting_params.verbosity, 
                                   random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs,
                                   max_depth = pm_config.max_depth,
                                   min_child_weight = pm_config.min_child_weight,
                                   gamma = pm_config.gamma,
                                   subsample = pm_config.subsample,
                                   colsample_bytree = pm_config.colsample_bytree,
                                   eta = pm_config.eta,
                                   n_estimators = pm_config.n_estimators,
                                   reg_alpha = pm_config.reg_alpha,
                                   reg_lambda = pm_config.reg_lambda)
    
    fit_params={
        "base__eval_set" : [[X_val, y_val]],
        # "base__eval_metric" : pm_config.setting_params.eval_metric,
        # "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds
        }
    
    if ibp is None:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("base", base_model)
        ])
    else:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("imbalance",ibp),
            ("base", base_model)
        ])
    
    return pipe, model_name, fit_params

def lrc_construction(pm_config, X_val = None, y_val= None, n_jobs = 1):
    print('using lr')
    model_name="lrc_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler()
    #imbalance processing
    if pm_config.setting_params.imb_type == "ros":
        ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    elif pm_config.setting_params.imb_type == "rus":
        ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    else:
        ibp = None
    #base model
    base_model = LogisticRegression(random_state=pm_config.setting_params.random_state)
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    hyper_parameters = { 
        "base__C": pm_config.range_params.C,
        "base__solver": pm_config.range_params.solver,
        "base__penalty": [ "l2"]
    }
    
    fit_params = {}
    
    if ibp is None:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("base", base_model)
        ])
    else:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("imbalance",ibp),
            ("base", base_model)
        ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, scoring="roc_auc", verbose=pm_config.setting_params.cv_verbose, cv = pm_config.setting_params.cv_fold, n_jobs=n_jobs, error_score='raise')
    return clf, model_name, fit_params

def lrc_finalize(pm_config, X_val = None, y_val= None):
    model_name="lrc_final_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler()
    #imbalance processing
    if pm_config.setting_params.imb_type == "ros":
        ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    elif pm_config.setting_params.imb_type == "rus":
        ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    else:
        ibp = None
    #base model
    base_model = LogisticRegression(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs,
                                   C = pm_config.C,
                                   solver = pm_config.solver,
                                   penalty =pm_config.penalty
    )
    
    fit_params = {}
    
    if ibp is None:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("base", base_model)
        ])
    else:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("imbalance",ibp),
            ("base", base_model)
        ])
    
    return pipe, model_name, fit_params
  
def rfc_construction(pm_config, X_val = None, y_val= None, n_jobs = 1):
    model_name="rfc_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler()
    #imbalance processing
    if pm_config.setting_params.imb_type == "ros":
        ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    elif pm_config.setting_params.imb_type == "rus":
        ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    else:
        ibp = None
    #imbalanced data processing
    base_model = RandomForestClassifier(random_state=pm_config.setting_params.random_state)
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    hyper_parameters = {
        'base__bootstrap': pm_config.range_params.bootstrap,
        'base__max_depth': pm_config.range_params.max_depth,
        'base__max_features': pm_config.range_params.max_features,
        'base__min_samples_leaf': pm_config.range_params.min_samples_leaf,
        'base__min_samples_split': pm_config.range_params.min_samples_split,
        'base__n_estimators': pm_config.range_params.n_estimators
        
    }
    
    fit_params = {}
    
    if ibp is None:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("base", base_model)
        ])
    else:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("imbalance",ibp),
            ("base", base_model)
        ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, scoring="roc_auc", verbose=pm_config.setting_params.cv_verbose, cv = pm_config.setting_params.cv_fold, n_jobs=n_jobs)
    return clf, model_name, fit_params

def rfc_finalize(pm_config, X_val = None, y_val= None):
    model_name="rfc_final_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler()
    #imbalance processing
    if pm_config.setting_params.imb_type == "ros":
        ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    elif pm_config.setting_params.imb_type == "rus":
        ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    else:
        ibp = None
    #base model
    base_model = RandomForestClassifier(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs,
                                   bootstrap = pm_config.bootstrap,
                                   max_depth = pm_config.max_depth,
                                   max_features = pm_config.max_features,
                                   min_samples_leaf = pm_config.min_samples_leaf,
                                   min_samples_split = pm_config.min_samples_split,
                                   n_estimators = pm_config.n_estimators,)
    
    fit_params = {}
    
    if ibp is None:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("base", base_model)
        ])
    else:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("imbalance",ibp),
            ("base", base_model)
        ])
    
    return pipe, model_name, fit_params    
def mlp_construction(pm_config, X_val=None, y_val=None, n_jobs=1):
    print('* using mlp')
    model_name = "mlp_"
    
    # Imputation
    simim = SimpleImputer(strategy="mean")
    
    # Preprocessing
    scaler = MinMaxScaler()
    
    # Imbalance processing
    if pm_config.setting_params.imb_type == "ros":
        ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    elif pm_config.setting_params.imb_type == "rus":
        ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    else:
        ibp = None
    
    base_model = MLPClassifier(random_state=pm_config.setting_params.random_state)
    
    hyper_parameters = {
        "base__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "base__activation": ['relu', 'tanh'],
        "base__solver": ['adam', 'sgd'],
        "base__alpha": [0.0001, 0.001],
        "base__learning_rate": ['constant', 'adaptive'],
        # previous 200,400
        "base__max_iter": [100,500,1000]
    }
    
    fit_params = {
    }
    
    if ibp is None:
        pipe = Pipeline([
            ("imputation", simim),
            ("scaler", scaler),
            ("base", base_model)
        ])
    else:
        pipe = Pipeline([
            ("imputation", simim),
            ("scaler", scaler),
            ("imbalance", ibp),
            ("base", base_model)
        ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, scoring="roc_auc_ovr", verbose=pm_config.setting_params.cv_verbose, cv=pm_config.setting_params.cv_fold, n_jobs=n_jobs, error_score='raise')
    return clf, model_name, fit_params

def mlp_finalize(pm_config, X_val = None, y_val= None):
    model_name="mlp_final_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler()
    #imbalance processing
    if pm_config.setting_params.imb_type == "ros":
        ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    elif pm_config.setting_params.imb_type == "rus":
        ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    else:
        ibp = None
    #base model
    base_model = MLPClassifier(random_state=pm_config.setting_params.random_state)
    
    fit_params = {}
    
    if ibp is None:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("base", base_model)
        ])
    else:
        pipe = Pipeline([
            ("imputation",simim),
            ("scaler",scaler),
            ("imbalance",ibp),
            ("base", base_model)
        ])
    
    return pipe, model_name, fit_params 