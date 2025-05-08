import numpy as np
import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

def xgb_construction(X_val = None, y_val= None):
    model_name="xgb_HYPER_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=42)
    #ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = xgb.XGBClassifier(verbosity = 0, random_state=42)
    # feature selection
    selector = RFE(base_model,verbose=1)
    
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
                        "base__max_depth": [7, 9, 11,13, 15],
                        "base__min_child_weight":[0.05, 0.1, 0.3 , 0.7, 1.5],
                        "base__gamma": [15, 17, 19, 21, 23, 25],
                        'base__subsample':[0.55, 0.6, 0.7, 0.8, 0.9],
                        'base__colsample_bytree':[0.7, 0.8, 0.9],
                        "base__eta" : [0.05, 0.1, 0.2],
                        "base__n_estimators": [200, 250, 300]
                       }
    
    fit_params={"base__early_stopping_rounds":10, 
            "base__eval_metric" : "auc", 
            "base__eval_set" : [[X_val, y_val]]}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        #("selector", selector),
        ("base", base_model)
    ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters,verbose=10,cv = 5,n_jobs=60)
    return clf, model_name, fit_params

def xgb_finalize(X_val = None, y_val= None):
    model_name="xgb_FINAL_SMT_"
    #imputation
    simim = SimpleImputer(strategy="mean")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=42)
    # ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = xgb.XGBClassifier(verbosity = 1, random_state=42, n_jobs=-1,
                                  max_depth = 13,
                                   min_child_weight=0.05,
                                   gamma = 15,
                                   subsample = 0.9,
                                   colsample_bytree = 0.7,
                                   eta = 0.02,
                                   n_estimators = 200)
    
    fit_params={"base__early_stopping_rounds":10, 
            "base__eval_metric" : "auc", 
            "base__eval_set" : [[X_val, y_val]]}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    return pipe, model_name, fit_params


def mlp_construction():
    model_name="mlp_ROS_"
    #imputation
    simim = SimpleImputer(strategy="median")
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    #ibp = SMOTE(random_state=42)
    ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = MLPClassifier(verbose = 0, random_state=42)
    # feature selection
    selector = RFE(base_model,verbose=1)
        
    hyper_parameters = { "imputation__strategy": ["median","mean"],}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        #("selector", selector),
        ("base", base_model)
    ])
    
    # CV grid search
    #clf = GridSearchCV(pipe, hyper_parameters,verbose=1,cv = 5)
    
    return pipe, model_name

def lr_construction():
    model_name="lrc_ROS_"
    #imputation
    simim = SimpleImputer()
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    #ibp = SMOTE(random_state=42)
    ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = LogisticRegression(solver="liblinear", tol=0.1)
    # feature selection
    selector = RFE(base_model,verbose=1)
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    hyper_parameters = { "base__C": [0.5,1,1.5],}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        #('select', selector),
        ("base", base_model)
    ])
    
    # CV grid search
    #clf = GridSearchCV(pipe, hyper_parameters,verbose=1,cv = 5)
    
    return pipe, model_name

def lasso_construction():
    model_name="lasso_FS_SMT_"
    #imputation
    simim = SimpleImputer()
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=42)
    #ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = LogisticRegression(solver="liblinear", tol=0.1, penalty="l1")
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    hyper_parameters = { 'max_iter': [1000], 
                            'C': np.arange(0.05, 0.1, 0.02),
                            'solver': ['saga']}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    #clf = GridSearchCV(pipe, hyper_parameters,verbose=1,cv = 5)
    
    return pipe, model_name


def gbm_construction():
    model_name="gbm_ORI_"
    #imputation
    simim = SimpleImputer()
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    #ibp = SMOTE(random_state=42)
    ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = GradientBoostingClassifier(verbose=1)
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    hyper_parameters = { 'max_iter': [1000], 
                            'C': np.arange(0.05, 0.1, 0.02),
                            'solver': ['saga']}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        #("ibp", ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    #clf = GridSearchCV(pipe, hyper_parameters,verbose=1,cv = 5)
    
    return pipe, model_name

def rfc_construction():
    model_name="rfc_SMT_"
    #imputation
    simim = SimpleImputer()
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=42)
    #ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = RandomForestClassifier(verbose=1)
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    hyper_parameters = {'base__bootstrap': [True, False],
                         'base__max_depth': [3, 5, 7, None],
                         'base__max_features': ['auto', 'sqrt'],
                        'base__min_samples_leaf': [1, 2, 4],
                        'base__min_samples_split': [2, 5, 10],
                         'base__n_estimators': [200, 400, 600]}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    clf = RandomizedSearchCV(pipe, hyper_parameters,verbose=1,cv = 5, n_jobs=-1)
    
    return pipe, model_name


def svc_construction():
    model_name="svc_ROS_"
    #imputation
    simim = SimpleImputer()
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    #ibp = SMOTE(random_state=42)
    ibp = RandomOverSampler(random_state=42)
    #base model
    base_model = SVC(verbose=1,probability=True,kernel = 'linear')
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    hyper_parameters = { 'max_iter': [1000], 
                            'C': np.arange(0.05, 0.1, 0.02),
                            'solver': ['saga']}
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    #clf = GridSearchCV(pipe, hyper_parameters,verbose=1,cv = 5)
    
    return pipe, model_name


def prep_construction():
    #imputation
    simim = SimpleImputer()
    #preprocessing
    scaler = MinMaxScaler() #MaxAbsScaler() 
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
    ])
    
    return pipe

def imb_construction():
    #imbalanced data processing
    ibp = SMOTE(random_state=42)
    #ibp = RandomOverSampler(random_state=42)
    
    return ibp
    
def fml_xgb(pm_config, X_val = None, y_val= None):
    model_name = "fml_xgb"
    
    base_model = xgb.XGBClassifier(verbosity = pm_config.verbosity, 
                                   random_state = pm_config.random_state, 
                                   n_jobs = pm_config.n_jobs,
                                   max_depth = pm_config.max_depth,
                                   min_child_weight = pm_config.min_child_weight,
                                   gamma = pm_config.gamma,
                                   subsample = pm_config.subsample,
                                   colsample_bytree = pm_config.colsample_bytree,
                                   eta = pm_config.eta,
                                   n_estimators = pm_config.n_estimators,
                                   eval_metric = pm_config.eval_metric,
                                   early_stopping_rounds = pm_config.early_stopping_rounds)
    
    fit_params={"eval_set" : [[X_val, y_val]]}
    
    return base_model, model_name, fit_params