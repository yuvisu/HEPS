import os
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def get_shap(all_model, X_test, save_path, shap_features_name, model_type, mapping):
    if model_type == 'xgboost_':
        explainer = shap.TreeExplainer(all_model.named_steps['base'], X_test.astype('float'))
        shap_values = explainer.shap_values(X_test.astype('float'))
    elif model_type == 'lrc_':
        explainer = shap.Explainer(all_model.named_steps['base'], X_test.astype('float'), seed=42)
        shap_values = explainer.shap_values(X_test.astype('float'))
    elif model_type == 'mlp_':
        background = shap.sample(X_test, int(np.floor(len(X_test)*0.1)))
        print('using background samples', len(background))
        explainer = shap.KernelExplainer(all_model.named_steps['base'].predict_proba, background.astype('float'), seed=42)
        shap_values = explainer.shap_values(background.astype('float'))
        X_test = background.astype('float')
        # explainer = shap.DeepExplainer(all_model.named_steps['base'], X_test.astype('float'))
        # shap_values = explainer.shap_values(X_test.astype('float'))


    if isinstance(shap_values, list):
        print(f'this shap value contains {len(shap_values)} elements.')
        for i in range(len(shap_values)):
            plot_shap(shap_values[i], X_test, shap_features_name, save_path, i)
    else:
        plot_shap(shap_values, X_test, shap_features_name, save_path, '2_classes')


def check_saving_path2(save_dir):
    
    isExist = os.path.exists(save_dir)
    if isExist is False: os.makedirs(save_dir)
    return save_dir


def plot_shap(shap_values_combined, X_test, shap_features_name, save_path, name):
    plt.figure(figsize=(15,9))
    shap.summary_plot(shap_values_combined, X_test.astype('float'), feature_names=shap_features_name, plot_type="dot", show=False, max_display=15)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.1e}'.format(x)))
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.4)
    plt.tight_layout()
    plt.xlabel('SHAP value', fontsize=13)
    # plt.title(f'SHAP Summary Plot', fontsize=23)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(save_path, f'shap_beeswarm_combined_{name}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    

    plt.figure(figsize=(15,9))
    shap.summary_plot(shap_values_combined, X_test.astype('float'), feature_names=shap_features_name, plot_type="bar", show=False, max_display=15)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.1e}'.format(x)))
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.4)
    plt.tight_layout()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2e') 
    plt.xlabel('mean(|SHAP value|)', fontsize=13)
    # plt.title(f'SHAP Bar Plot', fontsize=23)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(save_path, f'shap_bar_combined_{name}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    

def clean_raw_data(file):
    df = pd.read_csv(file)
    id = pd.DataFrame(df.ID.to_list(), columns = ['ID'])
    # print('original_df: ', df.shape)
    features_cols = ['age', 'race_ethnicity', 'sex',  'smoking',
       'ORIGINAL_BMI', 'SYSTOLIC', 'DIASTOLIC', 'hba1c', 'hdl',
       'Triglycerides', 'GGT', 'Obesity', 'Hypertension', 'Stroke',
       'Myocardial_Infarction', 'Ischemic_cardiovascular_disease',
       'Family_history_diabetes', 'CCI_flags', 'Alcohol', 'S1', 'S2', 'S3',
       'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14',
       'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24',
       'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'incidence_case',
       'op_enc_count_ehr', 'ip_enc_count_ehr', 'op_enc_count_claim',
       'ip_enc_count_claim', 'op_enc_count_ehr_all', 'ip_enc_count_ehr_all',
       'op_enc_count_claim_all', 'ip_enc_count_claim_all', 
       'mpec']
    label_cols = ['outcome', 'case_HSR_flag', 'case_CP1_flag', 'case_CP2_flag',
       'cp1+cp2+hsr']
    features = df[features_cols]
    num_cols = ['ORIGINAL_BMI',
    'SYSTOLIC',
    'DIASTOLIC',
    'hba1c',
    'hdl',
    'Triglycerides',
    'GGT']
    cat_cols = ['smoking']
    features.loc[:, num_cols] = features.loc[:, num_cols].apply(lambda col: col.fillna(0))
    features.loc[:, cat_cols] = features.loc[:, cat_cols].apply(lambda col: col.fillna(4))
    
    encoder = OneHotEncoder(sparse_output=False)
    encode_cols = ['race_ethnicity', 'sex']

    encoded_data = encoder.fit_transform(features[encode_cols])

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(encode_cols))
    # print('encoded_df', encoded_df.shape)

    features = pd.concat([features.drop(encode_cols, axis=1), encoded_df], axis=1)
    label = df[label_cols]
    # print(features.shape, label.shape)
    return features, label

def check_model_exist(root_dir, models_dir, model_name, model_id):
    # save_dir = os.path.join(root_dir, models_dir, model_name)
    # save_path = check_saving_path(save_dir,model_id)
    # path = Path(save_path)
    return False

def check_saving_path(save_dir,model_id):
    
    isExist = os.path.exists(save_dir)
    if isExist is False: os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_id)
    return save_path

def save_model(model, root_dir, models_dir, dataset_dir, model_name, year, model_id):
    save_dir = os.path.join(root_dir, models_dir, dataset_dir, model_name, year)
    save_path = check_saving_path(save_dir,model_id)
    pickle.dump(model, open(save_path, 'wb'))
    
def load_model(root_dir, models_dir, dataset_dir, model_name, file_name, model_id):
    save_dir = os.path.join(root_dir, models_dir, dataset_dir,model_name, file_name)
    isExist = os.path.exists(save_dir)
    if isExist is False: 
        print("No folder exist!!!")
        return None
    save_path = os.path.join(save_dir, model_id)
    print(f'* loading model: {save_path}')
    loaded_model = pickle.load(open(save_path, 'rb'))
    return loaded_model

def save_dataframe(dataframe, root_dir, models_dir, dataset, model_name,file, model_id):
    save_dir = os.path.join(root_dir, models_dir,  dataset, model_name,file)
    save_path = check_saving_path(save_dir, model_id)
    dataframe.to_csv(save_path)
    
def save_params_as_json(params, save_dir, file_name, model_alg, model_name, model_id):
    save_dir = os.path.join(save_dir, file_name, model_alg, model_name)
    save_path = check_saving_path(save_dir, model_id+".json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    
## can be removed
def save_fairness(dataframe, root_dir, output_fariness, model_name, model_id, filename):
    save_dir =  os.path.join(root_dir, output_fariness, model_name, model_id)
    isExist = os.path.exists(save_dir)
    if isExist is False: os.makedirs(save_dir)
    save_path = os.path.join(root_dir, output_fariness, model_name, model_id, filename)
    dataframe.to_csv(save_path)
    