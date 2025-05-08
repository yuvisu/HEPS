import os
import sys
import csv
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import shap as shap_module

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
    os.makedirs('statistic', exist_ok=True)
    for mpec_test in ['whole', 'smaller_threshold', 'larger_threshold']:
        print(f'******procssing_{mpec_type}')
        if mpec_test == 'whole':
            temp_features, temp_label = features, label
            pd.concat([temp_features, temp_label], axis = 1).drop(columns=['mpec']).to_csv(f'statistic/{y}year_{current_lable}_l+h.csv')
        elif mpec_test == 'smaller_threshold':
            temp_features = features[features.mpec <= 0.3]
            temp_label = label.iloc[temp_features.index]
            pd.concat([temp_features, temp_label], axis = 1).drop(columns=['mpec']).to_csv(f'statistic/{y}year_{current_lable}_l.csv')
        else:
            temp_features = features[features.mpec > 0.3]
            temp_label = label.iloc[temp_features.index]
            pd.concat([temp_features, temp_label], axis = 1).drop(columns=['mpec']).to_csv(f'statistic/{y}year_{current_lable}_h.csv')
        temp_features, temp_label = temp_features.reset_index(drop=True).drop(columns=['mpec']), temp_label.reset_index(drop=True)
        target = temp_label[current_lable]
    
        print('original target: ', target.value_counts())
        less = target[target == target.value_counts().sort_values(ascending=True).index.to_list()[0]]
        more = target[target == target.value_counts().sort_values(ascending=True).index.to_list()[1]].sample(n=len(less), random_state=42)
        Y = pd.concat([less,more])
        X = temp_features.iloc[Y.index]
        original_cols = X.columns
        print('y shape: ', Y.shape)
        print(Y.value_counts())
        print('x shape: ', X.shape)

        X_dict[mpec_test] = X
        Y_dict[mpec_test] = Y

    train_auroc_list = []
    test_auroc_list1 = []
    test_auroc_list2 = []
    test_auroc_list3 = []
    # _, _, _, test_count = train_test_split(X_dict['larger_threshold'],Y_dict['larger_threshold'], stratify=Y_dict['larger_threshold'], test_size=mparam.setting_params.train_test_ratio, random_state=42) 
    downsampling_count = len(Y_dict['larger_threshold'])

    X_all_dict = {}
    Y_all_dict = {}  
    # downsampling
    for mp in ['whole', 'smaller_threshold']:
        downsample_ratio = downsampling_count / len(Y_dict[mp])
        downsampled_data, _ = train_test_split(Y_dict[mp], 
                                                train_size=downsample_ratio, 
                                                stratify=Y_dict[mp], 
                                                random_state=42)
        pd.concat([X_dict[mp].loc[downsampled_data.index].reset_index(drop=True), Y_dict[mp].loc[downsampled_data.index].reset_index(drop=True)], axis = 1).to_csv(f'statistic/{y}year_{current_lable}_{mp}_downsample.csv')
        X = X_dict[mp].loc[downsampled_data.index].reset_index(drop=True)
        Y = Y_dict[mp].loc[downsampled_data.index].reset_index(drop=True)
        X_all_dict[mp] = X
        Y_all_dict[mp] = Y


    X_all_dict['larger_threshold'] = X_dict['larger_threshold']
    Y_all_dict['larger_threshold'] = Y_dict['larger_threshold']
        
    for i in range(0,n_run):
        # print("==================================== iterate",i," running ========================")
        random_seed = i
        model_save_name = model_name + model_alg + model_id + "bootstrap/"+ str(i)+str(current_lable)

        x_group_dict = {}
        y_group_dict = {}
        
        #  not downsample
        for mp in ['whole', 'smaller_threshold', 'larger_threshold']:
            #Generate Training and Testing Set
            X_train, X_test, y_train, y_test = train_test_split(X_dict[mp], Y_dict[mp], stratify=Y_dict[mp], test_size=mparam.setting_params.train_test_ratio, random_state=random_seed) 
            #Generate Training and Evaluation Set
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=mparam.setting_params.train_val_ratio, random_state=random_seed) #0.125 * 0.8 = 0.1
            x_group_dict[mp] = [X_train, X_val, X_test]
            y_group_dict[mp] = [y_train, y_val, y_test]

        X_train = x_group_dict[mpec_type][0].to_numpy()
        X_val = x_group_dict[mpec_type][1].to_numpy()
        y_train = y_group_dict[mpec_type][0].to_numpy()
        y_val = y_group_dict[mpec_type][1].to_numpy()


        y_train_count = len(y_train)


        print(X_train.shape, X_test.shape, X_val.shape)

        clf, _, fit_params = model_finalize(mparam, X_val=X_val, y_val=y_val)
        clf.fit(X_train, y_train, **fit_params)
        save_model(clf, root_dir, models_dir, os.path.join(dataset, mpec_type), cfg.model_name+'bootstrap', os.path.join((str(y)+'year'), str(current_lable)),'bootstrap_'+str(i)+"_clf.pk")
        for mpec_test in ['whole', 'smaller_threshold', 'larger_threshold']:
            X_test = x_group_dict[mpec_test][2].to_numpy()
            y_test = y_group_dict[mpec_test][2].to_numpy()

            X_test = pd.DataFrame(X_test).fillna(pd.DataFrame(X_test).mean()).values
            
            if mpec_test == 'whole':
                _, _,_, y_orig = train_test_split(X_dict['whole'], Y_dict['whole'], stratify=Y_dict['whole'], test_size=mparam.setting_params.train_test_ratio, random_state=random_seed) 
                original_y_test_count = len(y_orig)
                downsample_y_test_count = len(y_test)
                mpec_test_whole = [y_train_count, original_y_test_count, downsample_y_test_count]
            if mpec_test == 'smaller_threshold':
                _, _,_, y_orig = train_test_split(X_dict['smaller_threshold'], Y_dict['smaller_threshold'], stratify=Y_dict['smaller_threshold'], test_size=mparam.setting_params.train_test_ratio, random_state=random_seed) 
                original_y_test_count = len(y_orig)
                downsample_y_test_count = len(y_test)
                mpec_test_smaller_threshold = [y_train_count, original_y_test_count, downsample_y_test_count]
            if mpec_test == 'larger_threshold':
                _, _,_, y_orig = train_test_split(X_dict['larger_threshold'], Y_dict['larger_threshold'], stratify=Y_dict['larger_threshold'], test_size=mparam.setting_params.train_test_ratio, random_state=random_seed) 
                original_y_test_count = len(y_orig)
                downsample_y_test_count = len(y_test)
                mpec_test_larger_threshold = [y_train_count, original_y_test_count, downsample_y_test_count]
            print( X_test.shape)
            
            if random_seed == cfg.setting_params.random_state:
                """
                shap for statistic
                """
                shap_root = 'statistic/shap_value'
                os.makedirs(shap_root, exist_ok=True)
                X_test_preprocessed = clf.named_steps['imputation'].transform(X_test)
                X_test_preprocessed = clf.named_steps['scaler'].transform(X_test_preprocessed)
                if model_name == 'xgboost':
                    explainer = shap_module.TreeExplainer(clf.named_steps['base'], X_test_preprocessed.astype('float'))
                    print('train', X_train.shape)
                    print('test', X_test_preprocessed.shape)
                    shap_values = explainer.shap_values(X_test_preprocessed.astype('float'), check_additivity=False)
                    print('shape shap', shap_values.shape)
                    print('len shap', len(shap_values))
                    if len(shap_values) == 3 or len(shap_values) == 2:
                        for multi_shap in range(len(shap_values)):
                            print(shap_values[multi_shap])
                            shap_save = os.path.join(shap_root, f'shap_value_{dataset}_{mpec_type}_mpecTest_{mpec_test}{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_seed{i}_{multi_shap}.csv')
                            pd.DataFrame([shap_values[multi_shap]], columns=original_cols).to_csv(shap_save)
                    else:
                        shap_save = os.path.join(shap_root, f'shap_value_{dataset}_{mpec_type}_mpecTest_{mpec_test}{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_seed{i}.csv')
                        pd.DataFrame(shap_values, columns=original_cols).to_csv(shap_save)
                elif model_name == 'lrc':
                    explainer = shap_module.Explainer(clf.named_steps['base'], X_test_preprocessed.astype('float'), seed=42)
                    shap_values = explainer.shap_values(X_test_preprocessed.astype('float'))
                    if len(shap_values) == 3 or len(shap_values) == 2:
                        for multi_shap in range(len(shap_values)):
                            shap_save = os.path.join(shap_root, f'shap_value_{dataset}_{mpec_type}_mpecTest_{mpec_test}{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_seed{i}_{multi_shap}.csv')
                            pd.DataFrame(shap_values[multi_shap], columns=original_cols).to_csv(shap_save)
                    else:
                        shap_save = os.path.join(shap_root, f'shap_value_{dataset}_{mpec_type}_mpecTest_{mpec_test}{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_seed{i}.csv')
                        pd.DataFrame(shap_values, columns=original_cols).to_csv(shap_save)
                elif model_name == 'mlp':
                    background = shap_module.sample(X_test_preprocessed, int(np.floor(len(X_test_preprocessed)*0.1)))
                    explainer = shap_module.KernelExplainer(clf.named_steps['base'].predict_proba, background.astype('float'), seed=42)
                    shap_values = explainer.shap_values(background.astype('float'))
                    if len(shap_values) == 3 or len(shap_values) == 2:
                        for multi_shap in range(len(shap_values)):
                            shap_save = os.path.join(shap_root, f'shap_value_{dataset}_{mpec_type}_mpecTest_{mpec_test}{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_seed{i}_{multi_shap}.csv')
                            pd.DataFrame(shap_values[multi_shap], columns=original_cols).to_csv(shap_save)
                    else:
                        shap_save = os.path.join(shap_root, f'shap_value_{dataset}_{mpec_type}_mpecTest_{mpec_test}{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_seed{i}.csv')
                        pd.DataFrame(shap_values, columns=original_cols).to_csv(shap_save)
                        
            """
            auroc for statistic
            """
            test_pred = clf.predict(X_test)
            # have nan
            test_pred_prob = clf.predict_proba(X_test)
            temp_test = pd.concat([pd.DataFrame(np.nan_to_num(test_pred,0),columns=['predict']),pd.DataFrame(np.nan_to_num(y_test,0),columns=['true']),pd.DataFrame(np.nan_to_num(test_pred_prob,nan=0),columns=['prob_class_1', 'prob_class_2'])], axis = 1)
            if mpec_test == 'whole':
                performance_tab1.iloc[i] = performance_score(np.nan_to_num(y_test,0),np.nan_to_num(test_pred,0), np.nan_to_num(test_pred_prob[:, 1],nan=0))
                test_auroc_list1.append(temp_test)
            elif mpec_test == 'smaller_threshold':
                performance_tab2.iloc[i] = performance_score(np.nan_to_num(y_test,0),np.nan_to_num(test_pred,0), np.nan_to_num(test_pred_prob[:, 1],nan=0))
                test_auroc_list2.append(temp_test)
            elif mpec_test == 'larger_threshold':
                performance_tab3.iloc[i] = performance_score(np.nan_to_num(y_test,0),np.nan_to_num(test_pred,0), np.nan_to_num(test_pred_prob[:, 1],nan=0))
                test_auroc_list3.append(temp_test)

            if mpec_type == mpec_test:
                if mparam.setting_params.imb_type == 'Normal':
                    X_resampled, y_resampled = X_train, y_train
                    print("normal Resampled X:", X_resampled.shape)
                    print("normal Resampled y:", y_resampled.shape)
                else:
                    pipe_resample = clf[:3]
                    X_resampled, y_resampled = pipe_resample.fit_resample(X_train, y_train)

                    print("not normal Resampled X:", X_resampled.shape)
                    print("not normal Resampled y:", y_resampled.shape)
                X_resampled = pd.DataFrame(X_resampled).fillna(pd.DataFrame(X_resampled).mean()).values
                train_pred = clf.predict(X_resampled)
                train_pred_prob = clf.predict_proba(X_resampled)
                temp_train = pd.concat([pd.DataFrame(np.nan_to_num(train_pred,0),columns=['predict']),pd.DataFrame(np.nan_to_num(y_resampled,0),columns=['true']),pd.DataFrame(np.nan_to_num(train_pred_prob,nan=0),columns=['prob_class_1', 'prob_class_2'])], axis = 1)
                train_auroc_list.append(temp_train)


    auroc_root = 'statistic/auroc'
    os.makedirs(auroc_root, exist_ok=True)

    with open(os.path.join(auroc_root, f'train_predict_df_{dataset}_{mpec_type}_mpecTest_{mpec_test}_{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_100.pkl'), 'wb') as f:
        pickle.dump(train_auroc_list,f)

    with open(os.path.join(auroc_root, f'test_predict_df_{dataset}_{mpec_type}_mpecTest_whole_{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_100.pkl'), 'wb') as f:
        pickle.dump(test_auroc_list1,f)
    with open(os.path.join(auroc_root, f'test_predict_df_{dataset}_{mpec_type}_mpecTest_smaller_threshold_{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_100.pkl'), 'wb') as f:
        pickle.dump(test_auroc_list2,f)
    with open(os.path.join(auroc_root, f'test_predict_df_{dataset}_{mpec_type}_mpecTest_larger_threshold_{model_name}_{current_lable}_{year}year_{mparam.setting_params.imb_type}_100.pkl'), 'wb') as f:
        pickle.dump(test_auroc_list3,f)


    performance_tab1.loc[len(performance_tab1)] = performance_tab1.iloc[:, :-1].mean().round(4).values.tolist() + [np.array2string(np.array(mpec_test_whole))]
    save_dataframe(performance_tab1, root_dir, output_score_dir, os.path.join(dataset, mpec_type), cfg.model_name, str(y)+'year',f'{mpec_type}_mpecTest_whole_'+str(current_lable)+"_performance.csv" )
    
    performance_tab2.loc[len(performance_tab2)] = performance_tab2.iloc[:, :-1].mean().round(4).values.tolist() + [np.array2string(np.array(mpec_test_smaller_threshold))]
    save_dataframe(performance_tab2, root_dir, output_score_dir, os.path.join(dataset, mpec_type), cfg.model_name, str(y)+'year',f'{mpec_type}_mpecTest_smaller_threshold_'+str(current_lable)+"_performance.csv" )
    
    performance_tab3.loc[len(performance_tab3)] = performance_tab3.iloc[:, :-1].mean().round(4).values.tolist() + [np.array2string(np.array(mpec_test_larger_threshold))]
    save_dataframe(performance_tab3, root_dir, output_score_dir, os.path.join(dataset, mpec_type), cfg.model_name, str(y)+'year',f'{mpec_type}_mpecTest_larger_threshold_'+str(current_lable)+"_performance.csv" )
    save_model({"config":cfg, "param":mparam, "group":grp}, root_dir, models_dir, os.path.join(dataset, mpec_type), cfg.model_name+'bootstrap', str(y)+'year',str(current_lable)+"_experimental_config.pk")

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    
    parser.add_argument("--model_params", "-m", type=str, required=True)
    
    parser.add_argument("--group_info", "-g", type=str, required=True)

    args = parser.parse_args()
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    # for dataset in ['absolute_control', 'whole_control']:
    for dataset in ['absolute_control']:
        print(f'processing {dataset}')
        root = os.path.join(args.model_params, dataset)
        
        for mpec_type in os.listdir(root):
            print('*'*100,f'processing mpec_type {mpec_type}')
            root1 = os.path.join(root, mpec_type)
            for year in os.listdir(os.path.join(root1, cfg.model_name)):
                if year.endswith('year'):
                    print('*'*100,f'processing year {year}')
                    for outcome in os.listdir(os.path.join(root1, cfg.model_name, year)):
                        if outcome.endswith('.json'):
                            current_lable = outcome.split('label_')[1].split('_best')[0]
                            print('-'*50, f'processing outcome {current_lable}')
                            temp = os.path.join(root1, cfg.model_name, year, outcome)
                            print(f'*loading{temp}')
                            with open(args.setting) as json_file:
                                cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
                            
                            with open(temp) as json_file:
                                mparam = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
                                
                            with open(args.group_info) as json_file:
                                grp = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
                                
                            execute(cfg, mparam, grp, year, current_lable, dataset, mpec_type)
    print('done')