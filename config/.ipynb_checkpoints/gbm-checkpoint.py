from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

def gbm_construction():
    model_name="gbm"
    #imputation
    simim = SimpleImputer()
    #preprocessing
    scaler = MinMaxScaler() 
    #imbalanced data processing
    ibp = SMOTE(random_state=42)
    #base model
    base_model = GradientBoostingClassifier()
    
    # Parameters of pipelines can be set using '__' separated parameter names:
    hyper_parameters = { 'base__n_estimators':[130, 150, 170], 
                            'base__max_depth': [1,3,5],
                            'base__learning_rate': [0.01,0.1,1]}
    
    
    pipe = Pipeline([
        ("imputation",simim),
        ("scaler",scaler),
        ("ibp", ibp),
        ("base", base_model)
    ])
    
    # CV grid search
    clf = GridSearchCV(pipe, hyper_parameters, verbose=1, cv = 5)
    
    return clf, model_name