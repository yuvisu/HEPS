import os
import sys
import csv
import json
import shap
import warnings
import argparse
import numpy as np
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt

from types import SimpleNamespace

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold, train_test_split, KFold, RandomizedSearchCV

from config.model import model_construction
from util.io import *
import pickle

from joblib import Parallel, parallel_backend
from joblib import register_parallel_backend
from joblib import delayed
from joblib import cpu_count
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend



def execute(cfg,pfe):
    
    if cfg.parallel:
        FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(FILE_DIR)
        
        c = Client(profile=pfe)
        print("number of jobs = ",len(c))
        
        #c[:].map(os.chdir, [FILE_DIR]*len(c))
        
        bview = c.load_balanced_view()
        register_parallel_backend('ipyparallel', lambda : IPythonParallelBackend(view=bview))
    
    
    ''' Load settings '''
    model_name = cfg.model_name
    model_id = cfg.model_id
    model_alg = cfg.model_alg
    shap = cfg.shap
    
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    processed_dir = os.path.join(cfg.processed_dir, cfg.dataset_dir)

    output_roc_dir = cfg.output_roc_dir
    output_shap_dir = cfg.output_shap_dir
    output_score_dir = cfg.output_score_dir
    output_cv_dir = cfg.output_cv_dir
    
    ## full model
    process_dir = os.path.join(processed_dir)
    for i in os.listdir(process_dir):
        if i.endswith('year'):
            print('*'*100,f'processing {i} year dataset')
            year_dir = os.path.join(process_dir, i)
            for ii in os.listdir(year_dir):
                if ii.endswith('.csv'):
                    outcome = ii.split('target_')[1].split('.csv')[0]
                    print('\n\n','-'*50,f'* processing {outcome}')
                    features, label = clean_raw_data(os.path.join(year_dir, ii))
                    for mpec_type in ['whole', 'smaller_threshold', 'larger_threshold']:
                        print(f'******procssing_{mpec_type}')
                        print(f'outcome file{ii}')
                        if mpec_type == 'whole':
                            temp_features, temp_label = features, label
                        elif mpec_type == 'smaller_threshold':
                            temp_features = features[features.mpec <= 0.3]
                            temp_label = label.iloc[temp_features.index]
                        else:
                            temp_features = features[features.mpec > 0.3]
                            temp_label = label.iloc[temp_features.index]
                        temp_features, temp_label = temp_features.reset_index(drop=True).drop(columns=['mpec']), temp_label.reset_index(drop=True)
                        target = temp_label[outcome]
                        print('original features: ', temp_features.shape)
                        print('original features: ', temp_features.columns)
                        print('original labels: ', label.shape)
                        print('original labels: ', temp_label.columns)
                        # print('original target:\n', target.value_counts())
                        less = target[target == target.value_counts().sort_values(ascending=True).index.to_list()[0]]
                        more = target[target == target.value_counts().sort_values(ascending=True).index.to_list()[1]].sample(n=len(less), random_state=42)
                        Y = pd.concat([less,more])
                        X = temp_features.iloc[Y.index]
                        print('y shape: ', Y.shape)
                        print(Y.value_counts())
                        print('x shape: ', X.shape)
                        shap_features_name = X.columns.to_list()
                        
                        print('label', Y.isnull().sum())  
                        X = X.to_numpy()
                        Y = Y.to_numpy()

                        print('\n\n',  'split dataset\n\n')
                        print('random seed: ', cfg.setting_params.random_state)
                        #Generate Training and Testing Set
                        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=cfg.setting_params.train_test_ratio, random_state=cfg.setting_params.random_state) 
                        #Generate Training and Evaluation Set
                        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=cfg.setting_params.train_val_ratio, random_state=cfg.setting_params.random_state) #0.125 * 0.8 = 0.1
                        

                        print('\n\n', '*', 'tuning\n\n')
                        

                        # check the config file /fairness/config/model.py to setup more experimental models
                        if cfg.parallel:
                            clf, _, fit_params = model_construction(cfg, X_val=X_val, y_val=y_val, n_jobs = len(c))
                            with parallel_backend('ipyparallel'): 
                                all_model = clf.fit(X_train, y_train, **fit_params)
                        else:
                            print('X_train: ', X_train.shape, 'X_val: ', X_val.shape, 'X_test: ', X_test.shape)

                            clf, _, fit_params = model_construction(cfg, X_val=X_val, y_val=y_val, n_jobs = cfg.setting_params.n_jobs)

                            all_model = clf.fit(X_train, y_train,  **fit_params)

                            

                        # save the best model
                        print('saving output\n\n')
                        save_model(all_model, root_dir, models_dir, os.path.join(cfg.dataset_dir, mpec_type), model_name.split("_")[0], str(i),cfg.dataset_dir+'_'+model_name+'_'+str(i)+'year_'+str(outcome)+'.pkl')

                        save_dir = os.path.join(cfg.root_dir, cfg.output_cv_dir, os.path.join(cfg.dataset_dir, mpec_type), model_name.split("_")[0], str(i))
                        save_path = check_saving_path(save_dir, cfg.dataset_dir+'_'+ mpec_type+'_'+model_name+'_'+str(i)+'year_'+str(outcome)+cfg.output_cv_filename) 
                        df = pd.DataFrame(all_model.cv_results_)
                        df.to_csv(save_path)
                        if shap:
                            save_dir = os.path.join(cfg.root_dir, cfg.output_shap_dir, os.path.join(cfg.dataset_dir, mpec_type), model_name.split("_")[0], str(i),str(outcome))
                            save_path = check_saving_path2(save_dir)
                            best_pipeline = all_model.best_estimator_
                            X_test_preprocessed = best_pipeline.named_steps['imputation'].transform(X_test)
                            X_test_preprocessed = best_pipeline.named_steps['scaler'].transform(X_test_preprocessed)
                            get_shap(best_pipeline, X_test_preprocessed, save_path, shap_features_name, cfg.model_alg, {})                    
                        print('Done\n\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")
    
    args = parser.parse_args()
    
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    
    execute(cfg, args.profile)