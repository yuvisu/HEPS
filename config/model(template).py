import numpy as np
import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def model_construction(pm_config, X_val = None, y_val= None, n_jobs = 1):
    if pm_config.model_alg == "xgboost_":
        return xgb_construction(pm_config, X_val, y_val, n_jobs)
    elif pm_config.model_alg == "lrc_":
        return lr_construction(pm_config, X_val, y_val, n_jobs)
    elif pm_config.model_alg == "lasso_":
        return lasso_construction(pm_config, X_val, y_val, n_jobs)
    elif pm_config.model_alg == "rfc_":
        return rfc_construction(pm_config, X_val, y_val, n_jobs)
    elif pm_config.model_alg == "light_":
        return light_construction(pm_config, X_val, y_val, n_jobs)
    elif pm_config.model_alg == "cat_":
        return cat_construction(pm_config, X_val, y_val, n_jobs)
    else:
        return None

def model_finalize(pm_config, X_val = None, y_val= None):
    if pm_config.model_alg == "xgboost_":
        return xgb_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "lrc_":
        return lr_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "lasso_":
        return lasso_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "rfc_":
        return rfc_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "light_":
        return light_finalize(pm_config, X_val, y_val)
    elif pm_config.model_alg == "cat_":
        return cat_finalize(pm_config, X_val, y_val)
    else:
        return None

def model_fml(pm_config, X_val = None, y_val= None):
    if pm_config.model_alg == "xgboost_":
        return fml_xgb(pm_config, X_val, y_val)
    elif pm_config.model_alg == "lrc_":
        return fml_lrc(pm_config, X_val, y_val)
    else:
        return None
    
def prep_construction():
    #imputation
    simim = SimpleImputer()
    #preprocessing
    scaler = RobustScaler() #MaxAbsScaler() 
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
    ])
    
    return pipe

def imb_construction():
    #imbalanced data processing
    #ibp = SMOTE(random_state=42)
    ibp = RandomOverSampler(random_state=42)
    
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
                                   n_estimators = pm_config.n_estimators)
    
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
    model_name="xgb_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = SMOTE(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = xgb.XGBClassifier(verbosity = None, random_state=pm_config.setting_params.random_state)
    # feature selection
    selector = RFE(base_model,verbose=pm_config.setting_params.verbosity)
    
    # hyper_parameters = {
    #                     "base__max_depth": [x for x in range(7, 15)],
    #                     "base__min_child_weight": [x*0.1 for x in range(1, 15)],
    #                     "base__gamma": [x for x in range(15, 30, 2)],
    #                     'base__subsample':[x*0.01 for x in range(50, 95,5)],
    #                     'base__colsample_bytree':[x*0.01 for x in range(60, 95, 5)],
    #                     "base__eta" : [1/(10**x) for x in range(1, 5, 1)],
    #                     "base__n_estimators": [x for x in range(200, 500, 50)]
    #                    }

    hyper_parameters = {
                        "base__max_depth": pm_config.range_params.max_depth,
                        "base__min_child_weight":pm_config.range_params.min_child_weight,
                        "base__gamma": pm_config.range_params.gamma,
                        'base__subsample':pm_config.range_params.subsample,
                        'base__colsample_bytree':pm_config.range_params.colsample_bytree,
                        "base__eta" : pm_config.range_params.eta,
                        "base__n_estimators": pm_config.range_params.n_estimators,
                       }
    
    fit_params={
        "base__eval_metric": pm_config.setting_params.eval_metric,
        "base__early_stopping_rounds": pm_config.setting_params.early_stopping_rounds,
        "base__verbose": False,
        "base__eval_set": [[X_val, y_val]]}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("imbalance",ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, scoring="roc_auc", verbose=pm_config.setting_params.cv_verbose, cv = pm_config.setting_params.cv_fold, n_jobs=n_jobs)
    return clf, model_name, fit_params

def xgb_finalize(pm_config, X_val = None, y_val= None):
    model_name="xgb_FINAL_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
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
                                   n_estimators = pm_config.n_estimators)
    
    fit_params={
        "base__eval_set" : [[X_val, y_val]],
        "base__eval_metric" : pm_config.setting_params.eval_metric,
        "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("imbalance",ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params

def light_construction(pm_config, X_val = None, y_val= None, n_jobs = 1):
    model_name="light_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = LGBMClassifier(random_state=pm_config.setting_params.random_state)
    
    hyper_parameters = {
        "base__n_estimators": pm_config.range_params.n_estimators, # Number of boosted trees to fit.
        "base__learning_rate": pm_config.range_params.learning_rate, # Boosting learning rate
        "base__num_leaves": pm_config.range_params.num_leaves, # Maximum tree leaves for base learners.
        "base__max_depth": pm_config.range_params.max_depth, # Maximum tree depth for base learners
        "base__min_child_samples": pm_config.range_params.min_child_samples, # Minimum number of data needed in a child (leaf).
        "base__min_child_weight": pm_config.range_params.min_child_weight, # Minimum sum of instance weight (Hessian) needed in a child (leaf).
        "base__subsample": pm_config.range_params.subsample, #Subsample ratio of the training instance.
        "base__colsample_bytree": pm_config.range_params.colsample_bytree, # Subsample ratio of columns when constructing each tree.
        "base__reg_alpha": pm_config.range_params.reg_alpha, #L1 regularization term on weights.
        "base__reg_lambda": pm_config.range_params.reg_lambda # L2 regularization term on weights.
    }
    
    fit_params={
        "base__eval_metric" : pm_config.setting_params.eval_metric,
        "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds,
        "base__verbose" : False,
        "base__eval_set" : [[X_val, y_val]]}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("imbalance",ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, scoring="roc_auc", verbose=pm_config.setting_params.cv_verbose, cv = pm_config.setting_params.cv_fold, n_jobs=n_jobs)
    return clf, model_name, fit_params

def light_finalize(pm_config, X_val = None, y_val= None):
    model_name="light_FINAL_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = LGBMClassifier(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs,
                                   n_estimators = pm_config.n_estimators,
                                   learning_rate = pm_config.learning_rate,
                                   num_leaves = pm_config.num_leaves,
                                   max_depth = pm_config.max_depth,
                                   min_child_samples = pm_config.min_child_samples,
                                   min_child_weight = pm_config.min_child_weight,
                                   subsample = pm_config.subsample,
                                   colsample_bytree = pm_config.colsample_bytree,
                                   reg_alpha = pm_config.reg_alpha,
                                   reg_lambda = pm_config.reg_lambda)
    
    fit_params={
        "base__eval_set" : [[X_val, y_val]],
        "base__eval_metric" : pm_config.setting_params.eval_metric,
        "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("imbalance",ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params
    
def cat_construction(pm_config, X_val = None, y_val= None, n_jobs = 1):
    model_name="cat_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = CatBoostClassifier(random_state=pm_config.setting_params.random_state)
    
    hyper_parameters = {
        "base__depth" : pm_config.range_params.depth,
        "base__learning_rate" : pm_config.range_params.learning_rate,
        "base__iterations" : pm_config.range_params.iterations,
        "base__l2_leaf_reg": pm_config.range_params.l2_leaf_reg,
        "base__bagging_temperature": pm_config.range_params.bagging_temperature,
        "base__n_estimators": pm_config.range_params.n_estimators
    }
    
    fit_params={
        "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds,
        "base__verbose" : False,
        "base__eval_set" : (X_val, y_val)}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("imbalance",ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, scoring="roc_auc", verbose=pm_config.setting_params.cv_verbose, cv = pm_config.setting_params.cv_fold, n_jobs=n_jobs)
    return clf, model_name, fit_params

def cat_finalize(pm_config, X_val = None, y_val= None):
    model_name="cat_FINAL_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = CatBoostClassifier(
        random_state = pm_config.setting_params.random_state, 
        depth = pm_config.depth,
        learning_rate = pm_config.learning_rate,
        iterations = pm_config.iterations,
        l2_leaf_reg = pm_config.l2_leaf_reg,
        bagging_temperature = pm_config.bagging_temperature
        )
    
    fit_params={
        "base__eval_set" : (X_val, y_val),
        "base__early_stopping_rounds":pm_config.setting_params.early_stopping_rounds}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("imbalance",ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params
    
def lr_construction(pm_config, X_val = None, y_val= None, n_jobs = 1):
    model_name="lrc_ROS_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler()
    #imbalance processing
    ibp = RandomUnderSampler(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = LogisticRegression(random_state=pm_config.setting_params.random_state)
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    hyper_parameters = { 
        "base__C": pm_config.range_params.C,
        "base__solver": pm_config.range_params.solver,
        "base__penalty": pm_config.range_params.penalty
    }
    
    fit_params = {}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("imbalance",ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, scoring="f1", verbose=pm_config.setting_params.cv_verbose, cv = pm_config.setting_params.cv_fold, n_jobs=n_jobs)
    return clf, model_name, fit_params

def lr_finalize(pm_config, X_val = None, y_val= None):
    model_name="lrc_final_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = RobustScaler(quantile_range=(25, 75))
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = LogisticRegression(random_state = pm_config.setting_params.random_state, 
                                   n_jobs = pm_config.setting_params.n_jobs,
                                   C = pm_config.C,
                                   solver = pm_config.solver,
                                   penalty = pm_config.penalty)
    
    fit_params = {}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params
    
def lasso_construction(pm_config, X_val = None, y_val= None, n_jobs = 1):
    model_name="lasso_FS_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = LogisticRegression(
        penalty='l1', 
        random_state = pm_config.setting_params.random_state
        )
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    
    hyper_parameters = { 
        "base__C": pm_config.range_params.C,
        "base__solver": pm_config.range_params.solver
    }
    
    fit_params = {}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("imbalance",ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, scoring="roc_auc", verbose=pm_config.setting_params.cv_verbose, cv = pm_config.setting_params.cv_fold, n_jobs=n_jobs)
    return clf, model_name, fit_params

def lasso_finalize(pm_config, X_val = None, y_val= None):
    model_name="lasso_final_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
    #base model
    base_model = LogisticRegression(
        penalty='l1', 
        random_state = pm_config.setting_params.random_state,
        n_jobs = pm_config.setting_params.n_jobs,
        C = pm_config.C,
        solver = pm_config.solver)
    
    fit_params = {}
    
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
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
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
    scaler = MaxAbsScaler() 
    #imbalance processing
    ibp = RandomOverSampler(random_state=pm_config.setting_params.random_state)
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
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("imbalance",ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params    
    