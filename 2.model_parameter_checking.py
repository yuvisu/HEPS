import os
import sys
import csv
import json
import warnings
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from util.measure import performance_score

from util.io import check_saving_path, save_model, load_model, save_params_as_json
from util.io import *

def execute(cfg):
    
    ''' Load settings '''
    model_name = cfg.model_name
    model_alg = cfg.model_alg
    model_id = cfg.model_id
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    
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
                        print('original target:\n', target.value_counts())
                        less = target[target == target.value_counts().sort_values(ascending=True).index.to_list()[0]]
                        more = target[target == target.value_counts().sort_values(ascending=True).index.to_list()[1]].sample(n=len(less), random_state=42)
                        Y = pd.concat([less,more])
                        X = temp_features.iloc[Y.index]
                        used_variables = X.columns.tolist()
                        print('y shape: ', Y.shape)
                        print(Y.value_counts())
                        print('x shape: ', X.shape)
                        shap_features_name = X.columns.to_list()
                        
                        X = X.to_numpy()
                        Y = Y.to_numpy()
 

                        #Generate Training and Testing Set
                        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=cfg.setting_params.train_test_ratio, random_state=cfg.setting_params.random_state) 
                        #Generate Training and Evaluation Set
                        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=cfg.setting_params.train_val_ratio, random_state=cfg.setting_params.random_state) #0.125 * 0.8 = 0.1
                        
                        model = load_model(root_dir,  models_dir,  os.path.join(cfg.dataset_dir, mpec_type), model_name.split("_")[0], str(i),cfg.dataset_dir+'_'+model_name+'_'+str(i)+'year_'+str(outcome)+'.pkl')

                        #prediction
                        train_pred = model.predict(X_train)
                        train_pred_prob = model.predict_proba(X_train)

                        test_pred = model.predict(X_test)
                        test_pred_prob = model.predict_proba(X_test)

                        val_pred = model.predict(X_val)
                        val_pred_prob = model.predict_proba(X_val)

                        #create status index
                        train_flag_list = np.array(["train" for x in y_train])
                        test_flag_list = np.array(["test" for x in y_test])
                        val_flag_list = np.array(["val" for x in y_val])
                        
                        #format transformation
                        train_flag_list = np.reshape(train_flag_list, (-1, 1))
                        test_flag_list = np.reshape(test_flag_list, (-1, 1))
                        val_flag_list = np.reshape(val_flag_list, (-1, 1))
                        
                        train_score = np.reshape(train_pred_prob[:, 1], (-1,1))
                        test_score = np.reshape(test_pred_prob[:, 1], (-1,1))
                        val_score = np.reshape(val_pred_prob[:, 1], (-1,1))
                        
                        #data formation
                        train_label = np.reshape(y_train, (-1, 1))
                        train_with_outcome = np.concatenate((X_train, train_label), axis=1)
                        train_with_score = np.concatenate((train_with_outcome, train_score), axis=1)
                        train_output = np.concatenate((train_with_score, train_flag_list), axis=1)
                        
                        test_label = np.reshape(y_test, (-1, 1))
                        test_with_outcome = np.concatenate((X_test, test_label), axis=1)
                        test_with_score = np.concatenate((test_with_outcome, test_score), axis=1)
                        test_output = np.concatenate((test_with_score, test_flag_list), axis=1)
                        
                        val_label = np.reshape(y_val, (-1, 1))
                        val_with_outcome = np.concatenate((X_val, val_label), axis=1)
                        val_with_score = np.concatenate((val_with_outcome, val_score), axis=1)
                        val_output = np.concatenate((val_with_score, val_flag_list), axis=1)
                        
                        #column name creation
                        used_variables.append("hospitalization")
                        used_variables.append(model_alg+"score")
                        used_variables.append("Train/Val/Test")
                        
                        #output df generation
                        train_output_df = pd.DataFrame(data=train_output, columns = used_variables)
                        test_output_df = pd.DataFrame(data=test_output, columns = used_variables)
                        val_output_df = pd.DataFrame(data=val_output, columns = used_variables)
                        
                        #output combination
                        total_output_df = train_output_df
                        total_output_df = pd.concat([total_output_df, val_output_df])
                        total_output_df = pd.concat([total_output_df, test_output_df])
                        
                        #total_output_df.to_csv(model_name + model_alg + model_id+".csv")
                        
                        performance = performance_score(y_test, test_pred, test_pred_prob[:, 1])
                        performance_on_training = performance_score(y_train, train_pred, train_pred_prob[:, 1])
                        
                        print("MODEL INFO: ", model_name + model_alg + model_id)
                        print("MODEL Settings: ", model)
                        print("MODEL Training Performance: ", performance_on_training)
                        print("MODEL Performance: ", performance)
                        print("GV BEST PARAMETERS: ", model.best_params_)
                        
                        save_params = dict()
                        
                        save_params['model_alg'] = model_alg
                        save_params['setting_params'] = cfg.setting_params.__dict__
                        
                        for (key, value) in model.best_params_.items():
                            save_params[key.split("__")[-1]] = value
                        
                        save_params_as_json(save_params,cfg.output_best_param_dir, os.path.join(cfg.dataset_dir, mpec_type), model_name.split("_")[0], str(i),cfg.dataset_dir+'_'+model_name+'_'+str(i)+'_label_'+str(outcome)+"_best")
                        
                        print("Save BEST PARAMETERS TO: ", cfg.output_best_param_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    args = parser.parse_args()
    
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    
    execute(cfg)