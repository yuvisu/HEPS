import os
import sys
import csv
import json
import warnings
import argparse
import numpy as np
import pandas as pd

from types import SimpleNamespace

from config.model import model_finalize
from util.io import *

from util.measure import performance_score, fariness_score
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

def execute(cfg, mparam, grp, year, current_lable, dataset, mpec_type):
    
    ''' Load settings '''
    model_name = cfg.model_name
    model_alg = cfg.model_alg
    model_id = cfg.model_id
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    processed_dir = cfg.processed_dir
    
    output_roc_dir = cfg.output_roc_dir
    output_shap_dir = cfg.output_shap_dir
    output_score_dir = cfg.output_score_dir
    n_run = cfg.n_run
    
    fairness_tab1 = pd.DataFrame(np.zeros((n_run, len(grp.fair_measure))))
    fairness_tab1.columns = grp.fair_measure
    fairness_tab2 = pd.DataFrame(np.zeros((n_run, len(grp.fair_measure))))
    fairness_tab2.columns = grp.fair_measure
    fairness_tab3 = pd.DataFrame(np.zeros((n_run, len(grp.fair_measure))))
    fairness_tab3.columns = grp.fair_measure

    performance_tab1 = pd.DataFrame(np.zeros((n_run, len(grp.perf_measure))))
    performance_tab1.columns = grp.perf_measure

    performance_tab2 = pd.DataFrame(np.zeros((n_run, len(grp.perf_measure))))
    performance_tab2.columns = grp.perf_measure

    performance_tab3 = pd.DataFrame(np.zeros((n_run, len(grp.perf_measure))))
    performance_tab3.columns = grp.perf_measure

    year_dir = os.path.join(processed_dir, dataset, year)
    y = year.split('_')[0]
    features, label = clean_raw_data(os.path.join(year_dir, f'match_cohort_baseline_{y}year_target_{current_lable}.csv'))
    X_dict = {}
    Y_dict = {}
    for mpec_test in ['whole', 'smaller_threshold', 'larger_threshold']:
        if mpec_test == 'whole':
            temp_features, temp_label = features, label
            print('temp shape',temp_features.shape)
        elif mpec_test == 'smaller_threshold':
            temp_features = features[features.mpec <= 0.3]
            print('temp shape',temp_features.shape)
            temp_label = label.iloc[temp_features.index]
        else:
            temp_features = features[features.mpec > 0.3]
            temp_label = label.iloc[temp_features.index]
            print('temp shape',temp_features.shape)
        temp_features, temp_label = temp_features.reset_index(drop=True).drop(columns=['mpec']), temp_label.reset_index(drop=True)
        target = temp_label[current_lable]
        print('original target: ', target.value_counts())
        less = target[target == target.value_counts().sort_values(ascending=True).index.to_list()[0]]
        more = target[target == target.value_counts().sort_values(ascending=True).index.to_list()[1]].sample(n=len(less), random_state=42)
        Y = pd.concat([less,more])
        X = temp_features.iloc[Y.index]
        print('y shape: ', Y.shape)
        print(Y.value_counts())
        print('x shape: ', X.shape)
        print('features x: ', X.columns)
        X_dict[mpec_test] = X
        Y_dict[mpec_test] = Y
        import pickle
        with open("data.pkl", "wb") as f:
            pickle.dump({"X_dict": X_dict, "Y_dict": Y_dict}, f)

   
    for i in range(0,n_run):
        print("==================================== iterate",i," running ========================")
        random_seed = i
        model_save_name = model_name + model_alg + model_id + "bootstrap/"+ str(i)
        
        #Generate Training and Testing Set
        X_train, X_test, y_train, y_test = train_test_split(X_dict[mpec_type], Y_dict[mpec_type], stratify=Y_dict[mpec_type], test_size=mparam.setting_params.train_test_ratio, random_state=random_seed) 
        #Generate Training and Evaluation Set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=mparam.setting_params.train_val_ratio, random_state=random_seed) #0.125 * 0.8 = 0.1
        clf = load_model(root_dir, os.path.join(models_dir, dataset, mpec_type), cfg.model_name+'bootstrap', year.replace("_",""), current_lable, f"bootstrap_{i}_clf.pk")
        print("Model Exists, Fairness Assessment Running !!!!!!!!!!!!")
        for mpec_test in ['whole', 'smaller_threshold', 'larger_threshold']:
            # set test set
            _, X_test, _, y_test = train_test_split(X_dict[mpec_test], Y_dict[mpec_test], stratify=Y_dict[mpec_test], test_size=mparam.setting_params.train_test_ratio, random_state=random_seed) 

            print( X_test.shape)       
            ''' Pre-run Settings '''
            if(grp.discrete):
                te_protected_group_idx = np.where(X_test[grp.protected_feature_name] == 1)
                te_privileged_group_idx = np.where(X_test[grp.privileged_feature_name] == 1) 
            else:
                te_protected_group_idx = np.where(X_test[grp.protected_feature_name] < grp.cutoff) 
                te_privileged_group_idx = np.where(X_test[grp.privileged_feature_name] >= grp.cutoff) 
                
            
            if (grp.is_mask_attr):
                X_train = X_train.drop(columns=grp.masked_attrs)
                X_test = X_test.drop(columns=grp.masked_attrs)
                X_val = X_val.drop(columns=grp.masked_attrs)
            
            print(X_train.shape, X_test.shape, X_val.shape)

            test_pred = clf.predict(X_test.to_numpy())
            test_pred_prob = clf.predict_proba(X_test.to_numpy())


            y_test = y_test.to_numpy()
            print(te_protected_group_idx)
            y_protected_test = y_test[te_protected_group_idx]
            y_privileged_test = y_test[te_privileged_group_idx]
            
            y_protected_pred = test_pred[te_protected_group_idx]
            y_privileged_pred = test_pred[te_privileged_group_idx]

            if len(y_protected_test) == 0 or len(y_privileged_test) == 0 or len(y_protected_pred) == 0 or len(y_privileged_pred) == 0:
                if(grp.discrete):
                    print(f'no left for this group: {grp.protected_feature_name}  {grp.privileged_feature_name}')
                else:
                    print(f'no left for this group: cut off {grp.cutoff}')
                if mpec_test == 'whole':
                    fairness_tab1.iloc[i] = [np.nan] * len(grp.fair_measure)
                elif mpec_test == 'smaller_threshold':
                    fairness_tab2.iloc[i] = [np.nan] * len(grp.fair_measure)
                elif mpec_test == 'larger_threshold':
                    fairness_tab3.iloc[i] = [np.nan] * len(grp.fair_measure)
            else:
                if mpec_test == 'whole':
                    fairness_tab1.iloc[i] = fariness_score(y_protected_test, y_privileged_test, y_protected_pred, y_privileged_pred)
                elif mpec_test == 'smaller_threshold':
                    fairness_tab2.iloc[i] = fariness_score(y_protected_test, y_privileged_test, y_protected_pred, y_privileged_pred)
                elif mpec_test == 'larger_threshold':
                    fairness_tab3.iloc[i] = fariness_score(y_protected_test, y_privileged_test, y_protected_pred, y_privileged_pred)
            
            test_pred_prob_filled = np.nan_to_num(test_pred_prob[:, 1], nan=0.0)
            if mpec_test == 'whole':
                performance_tab1.iloc[i] = performance_score(y_test, test_pred, test_pred_prob_filled)
            elif mpec_test == 'smaller_threshold':
                performance_tab2.iloc[i] = performance_score(y_test, test_pred, test_pred_prob_filled)
            elif mpec_test == 'larger_threshold':
                performance_tab3.iloc[i] = performance_score(y_test, test_pred, test_pred_prob_filled)    



    performance_tab1.loc[len(performance_tab1)] = performance_tab1.iloc[:, :-1].mean().round(4).values.tolist() + [None]
    fairness_tab1.loc[len(fairness_tab1)] = fairness_tab1.iloc[:, :-2].mean().round(4).values.tolist() + [None]*2

    save_dataframe(performance_tab1, root_dir, output_score_dir+'_fairness', os.path.join(dataset, mpec_type), cfg.model_name +'_fairness', str(y)+'year',str(current_lable)+f'mpec_{mpec_type}_Teston_whole_'+"_performance.csv" )
    save_dataframe(fairness_tab1, root_dir, output_score_dir+'_fairness', os.path.join(dataset, mpec_type), cfg.model_name +'_fairness', str(y)+'year',str(current_lable)+f'mpec_{mpec_type}_Teston_whole_'+f"_fairness_{grp.protected_feature_name}_{grp.privileged_feature_name}.csv" )


    performance_tab2.loc[len(performance_tab2)] = performance_tab2.iloc[:, :-1].mean().round(4).values.tolist() + [None]
    fairness_tab2.loc[len(fairness_tab2)] = fairness_tab2.iloc[:, :-2].mean().round(4).values.tolist() + [None]*2

    save_dataframe(performance_tab2, root_dir, output_score_dir+'_fairness', os.path.join(dataset, mpec_type), cfg.model_name +'_fairness', str(y)+'year',str(current_lable)+f'mpec_{mpec_type}_Teston_smaller_threshold_'+"_performance.csv" )
    save_dataframe(fairness_tab2, root_dir, output_score_dir+'_fairness', os.path.join(dataset, mpec_type), cfg.model_name +'_fairness', str(y)+'year',str(current_lable)+f'mpec_{mpec_type}_Teston_smaller_threshold_'+f"_fairness_{grp.protected_feature_name}_{grp.privileged_feature_name}.csv" )


    performance_tab3.loc[len(performance_tab3)] = performance_tab3.iloc[:, :-1].mean().round(4).values.tolist() + [None]
    fairness_tab3.loc[len(fairness_tab3)] = fairness_tab3.iloc[:, :-2].mean().round(4).values.tolist() + [None]*2

    save_dataframe(performance_tab3, root_dir, output_score_dir+'_fairness', os.path.join(dataset, mpec_type), cfg.model_name +'_fairness', str(y)+'year',str(current_lable)+f'mpec_{mpec_type}_Teston_larger_threshold_'+"_performance.csv" )
    save_dataframe(fairness_tab3, root_dir, output_score_dir+'_fairness', os.path.join(dataset, mpec_type), cfg.model_name +'_fairness', str(y)+'year',str(current_lable)+f'mpec_{mpec_type}_Teston_larger_threshold_'+f"_fairness_{grp.protected_feature_name}_{grp.privileged_feature_name}.csv" )
         
if __name__ == "__main__":

    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    parser.add_argument("--model_params", "-m", type=str, required=True)
    
    parser.add_argument("--group_info", "-g", type=str, required=True)
    
    args = parser.parse_args()
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    for dataset in ['absolute_control']:
        print(f'processing {dataset}')
        for mpec_type in ['whole', 'smaller_threshold', 'larger_threshold']:
            root = os.path.join(args.model_params, dataset, mpec_type, cfg.model_name)
            for year in os.listdir(root):
                if year.endswith('year'):
                    print('*'*100,f'processing year {year}')
                    for outcome in os.listdir(os.path.join(root, year)):
                        if outcome.endswith('.json'):
                            current_lable = outcome.split('label_')[1].split('_best')[0]
                            print('-'*50, f'processing outcome {current_lable}')
                            temp = os.path.join(root, year, outcome)
                            print(f'*loading{temp}')
                            with open(args.setting) as json_file:
                                cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
                            
                            with open(temp) as json_file:
                                mparam = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
                                
                            with open(args.group_info) as json_file:
                                grp = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
                                
                            execute(cfg, mparam, grp, year,  current_lable, dataset, mpec_type)
    print('done')